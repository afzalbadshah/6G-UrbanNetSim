"""End-to-end verification harness: packetizer + channel + shared-link + ledger.

This is a VERIFICATION harness (not the production PFE integration, which is a
Codex-owned task). It only composes public network + metrics APIs to prove the two
halves interoperate end to end and that the numbers are physically sensible.

Path exercised per transfer:
  FixedMtuPacketizer.packetize  ->  SixGChannelModel.evaluate (per packet)
  ->  SharedLink.reserve (FIFO transmission + queue + propagation)
  ->  TransmissionLedger.record / snapshot

Run as a script (`python -m tests.test_integration_channel_link`) to print a
cross-band comparison table, or under unittest for the assertions.
"""

import unittest

from config import radio as bands_config
from metrics.channel import SixGChannelModel
from network import (
    ChannelRequest,
    DeliveryStatus,
    Direction,
    FixedMtuPacketizer,
    LinkProfile,
    SharedLink,
    TrafficObject,
    TransmissionLedger,
    TransmissionOutcome,
)

PAYLOAD_BYTES = 50_000
TX_POWER_W = 10.0 ** ((23.0 - 30.0) / 10.0)  # 23 dBm


def make_profile(band: str) -> LinkProfile:
    return LinkProfile(
        profile_id=f"{band}-profile",
        technology="5G-NR" if band == "FR1" else "6G-NR",
        generation=bands_config.get_band(band)["generation"],
        band=band,
        network_mtu_bytes=1500,
        nominal_bit_rate_bps=1.0e9,
        propagation_speed_m_s=3.0e8,
        shared_resource_id=f"{band}-cell",
        channel_model_id="sixg",
    )


def run_transfer(band: str, distance_m: float, line_of_sight: bool = True):
    """Compose the full path for one application transfer; return a summary dict."""
    profile = make_profile(band)
    model = SixGChannelModel(profile)
    packetizer = FixedMtuPacketizer()
    link = SharedLink(profile)
    ledger = TransmissionLedger()

    traffic = TrafficObject(
        object_id=f"{band}-obj-1",
        flow_id="flow-1",
        source_id="device-1",
        destination_id="mec-1",
        created_at_s=0.0,
        payload_bytes=PAYLOAD_BYTES,
        direction=Direction.UPLINK,
    )
    result = packetizer.packetize(traffic, profile)

    per_values = []
    last_arrival_s = 0.0
    first_rate_bps = 0.0
    for i, packet in enumerate(result.packets):
        ch = model.evaluate(
            ChannelRequest(
                link_profile_id=profile.profile_id,
                hop_id="hop-1",
                requested_at_s=0.0,
                distance_m=distance_m,
                on_wire_bytes=packet.on_wire_bytes,
                direction=Direction.UPLINK,
                tx_power_w=TX_POWER_W,
                line_of_sight=line_of_sight,
            )
        )
        if i == 0:
            first_rate_bps = ch.achievable_rate_bps
        per_values.append(ch.packet_error_probability)

        if ch.outage:
            ledger.record(TransmissionOutcome(
                unit_id=packet.packet_id, hop_id="hop-1", status=DeliveryStatus.DROPPED,
                started_at_s=0.0, ended_at_s=0.0, attempts=1,
                payload_bytes=packet.payload_bytes, on_wire_bytes=packet.on_wire_bytes,
                useful_payload_bytes=0, failure_reason="channel_outage",
            ))
            continue

        reservation = link.reserve(
            requested_at_s=0.0,
            on_wire_bytes=packet.on_wire_bytes,
            achievable_rate_bps=ch.achievable_rate_bps,
            propagation_delay_s=ch.propagation_delay_s,
        )
        last_arrival_s = max(last_arrival_s, reservation.arrival_at_s)
        ledger.record(TransmissionOutcome(
            unit_id=packet.packet_id, hop_id="hop-1", status=DeliveryStatus.DELIVERED,
            started_at_s=reservation.transmission_started_at_s,
            ended_at_s=reservation.arrival_at_s, attempts=1,
            payload_bytes=packet.payload_bytes, on_wire_bytes=packet.on_wire_bytes,
            useful_payload_bytes=packet.payload_bytes,
        ))

    snap = ledger.snapshot()
    return {
        "band": band,
        "packets": len(result.packets),
        "rate_bps": first_rate_bps,
        "mean_per": sum(per_values) / len(per_values),
        "arrival_s": last_arrival_s,
        "snapshot": snap,
    }


class TestEndToEnd(unittest.TestCase):
    def test_byte_conservation(self):
        r = run_transfer("FR1", 100.0)
        snap = r["snapshot"]
        self.assertEqual(snap.offered_payload_bytes, PAYLOAD_BYTES)
        self.assertEqual(snap.useful_payload_bytes, PAYLOAD_BYTES)   # in-coverage: all delivered
        self.assertGreater(snap.on_wire_bytes, snap.offered_payload_bytes)  # headers add bytes

    def test_full_delivery_in_coverage(self):
        for band in ("FR1", "FR3", "mmWave", "subTHz"):
            with self.subTest(band=band):
                snap = run_transfer(band, 100.0)["snapshot"]
                self.assertEqual(snap.delivery_ratio, 1.0)

    def test_higher_band_higher_rate(self):
        rates = [run_transfer(b, 100.0)["rate_bps"]
                 for b in ("FR1", "FR3", "mmWave", "subTHz")]
        for a, b in zip(rates, rates[1:]):
            self.assertLess(a, b)

    def test_higher_rate_lower_transmission_time(self):
        fr1 = run_transfer("FR1", 100.0)["arrival_s"]
        subthz = run_transfer("subTHz", 100.0)["arrival_s"]
        self.assertLess(subthz, fr1)   # wider 6G band moves the same bytes faster

    def test_determinism(self):
        a = run_transfer("mmWave", 120.0)
        b = run_transfer("mmWave", 120.0)
        self.assertEqual(a["snapshot"], b["snapshot"])
        self.assertEqual(a["arrival_s"], b["arrival_s"])


def _print_table():
    print(f"\nEnd-to-end 6G path — {PAYLOAD_BYTES} B uplink @100 m, LoS\n")
    print(f"{'band':>8} {'pkts':>5} {'rate(Mbps)':>11} {'meanPER':>9} "
          f"{'arrival(ms)':>12} {'deliv':>6} {'wireEff':>8}")
    for band in ("FR1", "FR3", "mmWave", "subTHz"):
        r = run_transfer(band, 100.0)
        s = r["snapshot"]
        print(f"{band:>8} {r['packets']:>5} {r['rate_bps']/1e6:>11.1f} "
              f"{r['mean_per']:>9.2e} {r['arrival_s']*1e3:>12.3f} "
              f"{s.delivery_ratio:>6.2f} {s.wire_efficiency_ratio:>8.3f}")


if __name__ == "__main__":
    _print_table()
