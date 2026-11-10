"""Baseline packetization policies.

This module contains no channel, queue, retry, energy or cost behavior. Those
remain separate strategies owned by the future Packetization and Flow Engine.

How to extend: implement another ``Packetizer`` class in this module and inject
it into ``NetworkEngine``. Preserve payload/on-wire conservation and keep channel
or retry behavior out of packet segmentation.
"""

from __future__ import annotations

from math import ceil

from network.contracts import LinkProfile, PacketUnit, PacketizationResult, PDUSet, TrafficObject


class FixedMtuPacketizer:
    """Deterministic application-segmentation baseline for a configured MTU."""

    def packetize(self, traffic_object: TrafficObject, link_profile: LinkProfile) -> PacketizationResult:
        """Segment one application's bytes into deterministic MTU-sized packets."""

        max_payload_bytes = link_profile.max_payload_bytes
        packet_count = max(1, ceil(traffic_object.payload_bytes / max_payload_bytes))
        pdu_set_id = f"{traffic_object.object_id}:pdu:0"

        packets = []
        remaining_payload_bytes = traffic_object.payload_bytes
        headers = link_profile.header_stack
        for sequence_number in range(packet_count):
            payload_bytes = min(max_payload_bytes, remaining_payload_bytes)
            remaining_payload_bytes -= payload_bytes
            packets.append(
                PacketUnit(
                    packet_id=f"{pdu_set_id}:packet:{sequence_number}",
                    pdu_set_id=pdu_set_id,
                    sequence_number=sequence_number,
                    payload_bytes=payload_bytes,
                    created_at_s=traffic_object.created_at_s,
                    network_header_bytes=headers.network_header_bytes,
                    transport_header_bytes=headers.transport_header_bytes,
                    security_overhead_bytes=headers.security_overhead_bytes,
                    telemetry_overhead_bytes=headers.telemetry_overhead_bytes,
                    link_header_bytes=headers.link_header_bytes,
                    link_trailer_bytes=headers.link_trailer_bytes,
                    is_last=sequence_number == packet_count - 1,
                )
            )

        pdu_set = PDUSet(
            pdu_set_id=pdu_set_id,
            traffic_object_id=traffic_object.object_id,
            created_at_s=traffic_object.created_at_s,
            total_payload_bytes=traffic_object.payload_bytes,
            packet_count=packet_count,
            deadline_at_s=traffic_object.deadline_at_s,
            completion_rule=traffic_object.completion_rule,
            completion_threshold=traffic_object.completion_threshold,
            plane=traffic_object.plane,
            traffic_kind=traffic_object.traffic_kind,
            flow_role=traffic_object.flow_role,
            measurement_id=traffic_object.measurement_id,
            dependency_ids=traffic_object.dependency_ids,
        )
        return PacketizationResult(pdu_set=pdu_set, packets=tuple(packets))
