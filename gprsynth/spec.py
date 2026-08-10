"""Acquisition spec — the geometry every other module reads.

Round default numbers for a generic 24-channel road array, not the calibration of
any particular cart. They are deliberately plain — trace pitch on a real system is
an odometer constant with several decimals behind it, and shipping one survey's
calibration as a default would say more about that survey than about the model.
Nothing here depends on the exact values: replace them with your instrument's and
everything downstream follows.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Spec:
    nsample: int = 256          # samples per trace
    nchannel: int = 24          # cross-track channels
    dt_ns: float = 0.150        # sample interval [ns]
    d_channel_m: float = 0.080  # channel pitch [m]
    d_trace_m: float = 0.050    # along-track trace pitch [m]
    max_depth_m: float = 2.0    # depth window [m]

    @property
    def velocity_m_per_ns(self) -> float:
        """Two-way velocity implied by the depth window."""
        return 2 * self.max_depth_m / (self.nsample * self.dt_ns)

    def sample_of_depth(self, depth_m: float) -> float:
        return (2 * depth_m / self.velocity_m_per_ns) / self.dt_ns


DEFAULT = Spec()
