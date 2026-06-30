import zarr
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class GuianaDataloader(Dataset):
    """
    PyTorch Dataset for ocean simulation data stored in a Guiana zarr archive.

    Each sample is a contiguous sequence of `conditioning_steps + forecast_steps`
    frames spaced `dt` hours apart. The dataset iterates over all valid starting
    timestamps — those for which every frame in the full window exists in the archive.
    """

    def __init__(
        self,
        path,
        dt=24,
        transform=None,
        forecast_steps=1,
        conditioning_steps=3,
        start_time=None,
        end_time=None,
        time_range=None,
    ):
        """
        Args:
            path: Path to the zarr archive.
            dt: Time step between consecutive frames, in hours (also accepts pd.Timedelta or string).
            transform: Optional callable applied to the bundled tensor before unpacking channels.
            forecast_steps: Number of future frames the model should predict (target).
            conditioning_steps: Number of past frames provided as model input context.
            start_time: Only include samples whose first frame is on or after this time.
            end_time: Only include samples whose first frame is on or before this time.
            time_range: Alternative to (start_time, end_time) as a 2-element tuple.
        """
        self.path = path
        self.group = zarr.open_group(path, mode="r")
        self.tensor = self.group["ocean_tensor"]

        # Keep the raw zarr values so info()/invalid_times() can report exactly
        # which raw entries could not be parsed by pandas.
        self.raw_times = np.asarray(self.group["time"][:])
        self.times = pd.to_datetime(self.raw_times, errors="coerce")

        self.dt = self._timedelta(dt)
        self.transform = transform
        self.forecast_steps = int(forecast_steps)
        self.conditioning_steps = int(conditioning_steps)
        self.total_steps = self.conditioning_steps + self.forecast_steps
        self.start_time, self.end_time = self._parse_time_range(time_range, start_time, end_time)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.forecast_steps < 1:
            raise ValueError("forecast_steps must be >= 1")
        if self.conditioning_steps < 1:
            raise ValueError("conditioning_steps must be >= 1")

        self.channels = self._channels()
        self.bathymetry = self._array("bathymetry", "bat", "batim", "bathy")
        if self.bathymetry is None:
            raise ValueError("Bathymetry metadata was not found. Expected zarr array: bathymetry, bat, batim, or bathy")
        self.lat = self._array("lat", "latitude", "Latitude")
        self.lon = self._array("lon", "longitude", "Longitude")

        # Ignore invalid/NaT timestamps, then index by actual timestamp.
        self.time_to_idx = {}
        for i, t in enumerate(self.times):
            t = pd.Timestamp(t)
            if pd.isna(t):
                continue
            self.time_to_idx.setdefault(t, i)

        # Chronological order is kept because this is a time-series dataset, but
        # valid starts are selected only by timestamp membership. There is no
        # old-loader positional skip emulation here.
        self.sorted_times = sorted(self.time_to_idx)
        self.all_starts = [
            t
            for t in self.sorted_times
            if all(t + k * self.dt in self.time_to_idx for k in range(self.total_steps))
        ]
        self.starts = self._filter_starts(self.all_starts, self.start_time, self.end_time)

    def __len__(self):
        """Return the number of valid samples after applying the time-range filter."""
        return len(self.starts)

    def __getitem__(self, idx, start_time=None, end_time=None, time_range=None):
        """
        Get one sample, or create a time-filtered view.

        Normal usage:
            dataset[27]

        Time-filtered view, without loading tensors:
            dataset["2016-07-01":"2021-12-01"]
            dataset.time_range("2016-07-01", "2021-12-01")

        Single sample using an index relative to a time range:
            dataset[0, "2016-07-01", "2021-12-01"]
            dataset.__getitem__(0, start_time="2016-07-01", end_time="2021-12-01")
        """
        if isinstance(idx, tuple):
            if len(idx) == 2:
                return self.time_range(idx[0], idx[1])
            if len(idx) == 3:
                idx, start_time, end_time = idx
            else:
                raise IndexError("Use dataset[idx], dataset[start:stop], or dataset[idx, start, stop]")

        if isinstance(idx, slice) and self._looks_like_time_slice(idx):
            return self.time_range(idx.start, idx.stop)

        if start_time is not None or end_time is not None or time_range is not None:
            start, end = self._parse_time_range(time_range, start_time, end_time)
            starts = self._filter_starts(self.starts, start, end)
            if isinstance(idx, slice):
                return self._view(starts[idx], start, end)
            return self._get_from_start(starts[int(idx)])

        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        if isinstance(idx, np.integer):
            idx = int(idx)

        return self._get_from_start(self.starts[idx])

    def sample_times(self, idx):
        """Return the actual timestamps loaded for one item."""
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        return self._times_from_start(self.starts[int(idx)])

    def time_range(self, start_time=None, end_time=None, time_range=None):
        """Return a lightweight dataset view restricted by sample start time."""
        start, end = self._parse_time_range(time_range, start_time, end_time)
        return self._view(self._filter_starts(self.starts, start, end), start, end)

    def indices_in_time_range(self, start_time=None, end_time=None, time_range=None):
        """Return integer indices whose first loaded timestamp is inside the range."""
        start, end = self._parse_time_range(time_range, start_time, end_time)
        return [i for i, t in enumerate(self.starts) if self._in_time_range(t, start, end)]

    def info(self):
        """Return a metadata dict describing shape, time coverage, channels, and spatial extent."""
        h, w = self.tensor.shape[-2:]
        lat, lon = self._latlon_grid(h, w)
        valid_start = self.starts[0] if self.starts else None
        valid_end = self.sample_times(len(self.starts) - 1)[-1] if self.starts else None
        invalid_raw_times = self.invalid_times()

        return {
            "path": self.path,
            "tensor_shape": tuple(self.tensor.shape),
            "device": self.device,
            "time": {
                "raw_interval": (
                    [self._date(self.sorted_times[0]), self._date(self.sorted_times[-1])]
                    if self.sorted_times else None
                ),
                "native_dt": str(self._native_dt()),
                "sample_dt": str(self.dt),
                "sample_dt_hours": self.dt / pd.Timedelta(hours=1),
                "time_range_filter": [self._date(self.start_time), self._date(self.end_time)],
                "time_range_filter_applies_to": "first_loaded_frame",
                "conditioning_steps": self.conditioning_steps,
                "forecast_steps": self.forecast_steps,
                "total_steps_loaded": self.total_steps,
                "sample_span": str((self.total_steps - 1) * self.dt),
                "forecast_horizon_from_last_condition": str(self.forecast_steps * self.dt),
                "n_raw_times": len(self.times),
                "n_valid_raw_times": len(self.sorted_times),
                "n_invalid_raw_times": len(invalid_raw_times),
                "invalid_raw_time_indices": [x["index"] for x in invalid_raw_times],
                "invalid_raw_times": invalid_raw_times,
                "raw_time_dtype": str(self.raw_times.dtype),
                "n_valid_items_before_time_filter": len(self.all_starts),
                "n_valid_items": len(self.starts),
                "valid_interval": [self._date(valid_start), self._date(valid_end)] if valid_start is not None else None,
                "index_order": "chronological",
                "returned_dates": "final_loaded_frame",
                "valid_start_logic": "timestamp_membership",
            },
            "channels": {
                "loaded_names": ["u", "v", "ssh"],
                "loaded_indices": self.channels,
                "bathymetry_name": "bathymetry",
                "bathymetry_index": None,
                "bathymetry_source": "zarr_metadata_array",
                "bathymetry_sign_correction": 1,
                "available_names": self._channel_names(),
            },
            "space": {
                "height": h,
                "width": w,
                "latlon_source": "zarr" if self.lat is not None and self.lon is not None else "pixel_index",
                "lat_range": [float(np.nanmin(lat)), float(np.nanmax(lat))],
                "lon_range": [float(np.nanmin(lon)), float(np.nanmax(lon))],
                "top_left_latlon": [float(lat[0, 0]), float(lon[0, 0])],
                "top_right_latlon": [float(lat[0, -1]), float(lon[0, -1])],
                "bottom_left_latlon": [float(lat[-1, 0]), float(lon[-1, 0])],
                "bottom_right_latlon": [float(lat[-1, -1]), float(lon[-1, -1])],
                "center_latlon": [float(lat[h // 2, w // 2]), float(lon[h // 2, w // 2])],
            },
            "transform": None if self.transform is None else repr(self.transform),
        }

    def invalid_times(self):
        """Return raw zarr time entries that pandas could not parse."""
        bad = pd.isna(self.times)
        out = []

        for i in np.flatnonzero(bad):
            raw = self.raw_times[i]

            if isinstance(raw, bytes):
                raw_value = raw.decode(errors="replace")
            elif isinstance(raw, np.generic):
                raw_value = raw.item()
            else:
                raw_value = raw

            out.append({
                "index": int(i),
                "raw_value": str(raw_value),
                "raw_repr": repr(raw_value),
                "raw_type": type(raw_value).__name__,
            })

        return out

    def _get_from_start(self, t0):
        """
        Load and return one training sample whose sequence begins at timestamp t0.

        Returns:
            x     (Tensor [forecast_steps*C, H, W]): forecast target frames, stacked channel-first.
            meta  (dict): keys — y (conditioning frames), dates (year/month/day/hour of last frame),
                          latlon (window center), latlon_grid (full H×W grid), ssh (last conditioning SSH).
            mask  (Tensor [1, H, W]): 1 where the last conditioning frame contained NaN values.
        """
        ts = self._times_from_start(t0)
        arr = np.stack([np.asarray(self.tensor[self.time_to_idx[t]]) for t in ts])

        if arr.ndim == 5:
            arr = arr[:, 0]
        elif arr.ndim != 4:
            raise ValueError(f"Bad tensor shape: {arr.shape}")

        # Returned data channels stay limited to u/v/ssh; bathymetry is only
        # exposed separately as y["bat"].
        fields = arr[:, self.channels]
        bathymetry = self._bathymetry_from_array(arr)

        data = torch.tensor(np.nan_to_num(fields, nan=0.0), dtype=torch.float32)
        masks = torch.tensor(~np.isfinite(fields[:, :1]), dtype=torch.float32)

        h, w = data.shape[-2:]
        bat = torch.tensor(np.nan_to_num(bathymetry, nan=1.0), dtype=torch.float32)
        bat = bat.unsqueeze(0).unsqueeze(0).expand(self.total_steps, 1, h, w)

        lat, lon = self._latlon_grid(h, w)
        lat = torch.tensor(lat, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(self.total_steps, 1, h, w)
        lon = torch.tensor(lon, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(self.total_steps, 1, h, w)

        # Transform receives one tensor so crops stay aligned across time,
        # variables, mask, bathymetry, latitude, and longitude.
        bundle = torch.cat([masks, data, bat, lat, lon], dim=1)
        if self.transform is not None:
            bundle = self.transform(bundle)
            if not torch.is_tensor(bundle):
                bundle = torch.tensor(bundle, dtype=torch.float32)

        masks, data, bat, lat, lon = torch.split(bundle, [1, 3, 1, 1, 1], dim=1)
        bat = bat[-1]
        lat, lon = lat[-1, 0], lon[-1, 0]
        mask = masks[-1]

        h, w = data.shape[-2:]
        x = data[-self.forecast_steps:]        # frames the model must predict
        y_frames = data[:-self.forecast_steps]  # frames given as conditioning context
        ssh = y_frames[-1]                      # SSH snapshot of the last conditioning frame

        x = x.reshape(-1, h, w).to(self.device)
        y = y_frames.reshape(y_frames.size(1), y_frames.size(0), h, w).reshape(-1, h, w).to(self.device)

        # Date of the final loaded frame.
        t = ts[-1]
        dates = torch.tensor([t.year, t.month, t.day, t.hour]).to(self.device)

        # Match the legacy Guiana loader: lat/lon identify the crop center,
        # not the top-left corner.
        latlon = torch.stack([lat[h // 2, w // 2], lon[h // 2, w // 2]]).to(self.device)
        latlon_grid = torch.stack([lat, lon])

        return x, {
            "y": y,
            "dates": dates,
            "latlon": latlon,
            "latlon_grid": latlon_grid,
            "bat": bat.to(self.device),
            "ssh": ssh,
        }, mask.to(self.device)

    def _view(self, starts, start_time=None, end_time=None):
        """Create a shallow copy of this dataset restricted to the given start timestamps."""
        out = self.__class__.__new__(self.__class__)
        out.__dict__ = self.__dict__.copy()
        out.starts = list(starts)
        out.start_time = start_time
        out.end_time = end_time
        return out

    def _times_from_start(self, t0):
        """Return the list of `total_steps` timestamps starting at t0 and spaced by dt."""
        return [t0 + k * self.dt for k in range(self.total_steps)]

    def _parse_time_range(self, time_range, start_time, end_time):
        """Resolve (time_range, start_time, end_time) into a validated (start, end) Timestamp pair."""
        if time_range is not None:
            if start_time is not None or end_time is not None:
                raise ValueError("Use either time_range=(start, end) or start_time/end_time, not both")
            if len(time_range) != 2:
                raise ValueError("time_range must be a pair: (start_time, end_time)")
            start_time, end_time = time_range

        start = self._parse_time_bound(start_time, "start_time", is_end=False)
        end = self._parse_time_bound(end_time, "end_time", is_end=True)

        if start is not None and end is not None and start > end:
            raise ValueError("start_time must be <= end_time")

        return start, end

    def _parse_time_bound(self, value, name, is_end=False):
        """
        Parse a single time bound into a pd.Timestamp.

        Date-only end bounds (e.g. '2021-12-01') are expanded to end-of-day so
        that the full day is included. Exact datetimes are kept as-is.
        """
        if value is None:
            return None
        raw = value
        try:
            t = pd.Timestamp(value)
        except Exception as e:
            raise ValueError(f"{name} could not be parsed as a valid timestamp: {raw!r}") from e
        if pd.isna(t):
            raise ValueError(f"{name} could not be parsed as a valid timestamp: {raw!r}")

        # User convenience: an end bound like '2021-12-01' includes that whole day.
        # Exact datetimes such as '2021-12-01 00:00' are kept exact.
        if is_end and isinstance(raw, str) and self._is_date_only_string(raw):
            t = t + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        return t

    def _is_date_only_string(self, value):
        """Return True if value is a plain YYYY-MM-DD string with no time component."""
        value = value.strip()
        if len(value) != 10:
            return False
        return (
            value[4] == "-" and value[7] == "-"
            and value[:4].isdigit() and value[5:7].isdigit() and value[8:10].isdigit()
        )

    def _filter_starts(self, starts, start_time, end_time):
        """Return only those start timestamps that fall within [start_time, end_time]."""
        return [t for t in starts if self._in_time_range(t, start_time, end_time)]

    def _in_time_range(self, t, start_time=None, end_time=None):
        """Return True if timestamp t is within the optional [start_time, end_time] bounds."""
        if start_time is not None and t < start_time:
            return False
        if end_time is not None and t > end_time:
            return False
        return True

    def _looks_like_time_slice(self, s):
        """Return True if slice s uses non-integer bounds (treated as a time-based slice, not index-based)."""
        if s.step is not None:
            return False
        values = [x for x in (s.start, s.stop) if x is not None]
        if not values:
            return False
        return not all(isinstance(x, (int, np.integer)) for x in values)

    def _timedelta(self, value):
        """Convert an int/float (interpreted as hours), string, or pd.Timedelta into a pd.Timedelta."""
        if isinstance(value, pd.Timedelta):
            return value
        if isinstance(value, (int, float, np.integer, np.floating)):
            return pd.Timedelta(hours=value)
        return pd.Timedelta(value)

    def _native_dt(self):
        """Infer the most common time step between consecutive archived timestamps (modal gap)."""
        if len(self.sorted_times) < 2:
            return None
        deltas = pd.Series(self.sorted_times).diff().dropna()
        return deltas.mode().iloc[0] if len(deltas) else None

    def _date(self, t):
        """Convert a timestamp to an ISO string, or None if it is null."""
        if t is None or pd.isna(pd.Timestamp(t)):
            return None
        return str(pd.Timestamp(t))

    def _channels(self):
        """Return the zarr channel indices for [u, v, ssh], falling back to [0, 1, 2] if no metadata exists."""
        names = self._channel_names()
        if names is None:
            return [0, 1, 2]
        return [names.index(name) if name in names else i for i, name in enumerate(["u", "v", "ssh"])]

    def _channel_names(self):
        """Return the list of channel name strings from the zarr archive, or None if the key is absent."""
        if "channel" not in self.group:
            return None
        return [x.decode() if isinstance(x, bytes) else str(x) for x in self.group["channel"][:]]

    def _array(self, *names):
        """Return the first zarr array whose key matches any of the given name candidates, or None."""
        for name in names:
            if name in self.group:
                return np.asarray(self.group[name][:])
        return None

    def _bathymetry_from_array(self, arr):
        if self.bathymetry is not None:
            bat = self.bathymetry
            while bat.ndim > 2:
                bat = bat[0]
            return bat[:arr.shape[-2], :arr.shape[-1]]

        raise ValueError("Bathymetry metadata was not found")

    def _latlon_grid(self, h, w):
        """Return (lat, lon) 2-D grids of shape (h, w), using pixel indices as fallback if coordinates are absent."""
        lat = np.arange(h) if self.lat is None else self.lat
        lon = np.arange(w) if self.lon is None else self.lon
        if lat.ndim == 2 and lon.ndim == 2:
            return lat[:h, :w], lon[:h, :w]
        return np.meshgrid(lat[:h], lon[:w], indexing="ij")
