"""
Central citation registry for Claude-owned sourced config profiles.

The full bibliography (source name, URL, version) lives HERE, once — like a paper's
reference section — so the config modules stay clean and just cite a short id via
`cite(...)`. This keeps every default auditable (realism charter §11) without
scattering paper names/URLs through the code.

To add a source: add one entry to `_REFERENCES`. To cite it in a config file:
`cite("<ref_id>", maturity=..., validity_notes="...")`.
"""

from config._provenance import provenance

# ref_id -> bibliographic constants (the parts that never change per use).
_REFERENCES = {
    "sim_default": {
        "source_id": "SmartCityEdgeSim 1.2 config package",
        "source_version": "1.2",
    },
    "tr38901": {
        "source_id": "3GPP TR 38.901",
        "source_uri": "https://www.3gpp.org/DynaReport/38901.htm",
        "source_version": "Rel-17",
    },
    "3gpp_ue_power": {
        "source_id": "3GPP UE power class 3",
        "source_uri": "https://www.3gpp.org/",
        "source_version": "TS 38.101",
    },
    "att_flying_cow": {
        "source_id": "AT&T Flying COW (Cell on Wings)",
        "source_uri": "https://www.commercialuavnews.com/public-safety/at-t-s-flying-cow-transmits-5g-network-by-tethered-drone",
    },
    "stl_edge": {
        "source_id": "STL Partners, Edge data centres",
        "source_uri": "https://stlpartners.com/articles/edge-computing/edge-data-centres/",
    },
    "krishnan2021_uav": {
        "source_id": "Krishnan et al. 2021, Roofline Model for UAVs",
        "source_uri": "https://arxiv.org/pdf/2111.03792",
    },
    "green_grid_pue": {
        "source_id": "Uptime Institute / The Green Grid, PUE",
        "source_version": "PUE 2.x",
    },
    "eia_2024": {
        "source_id": "EIA Electric Power Monthly, Table 5.6.A",
        "source_uri": "https://www.eia.gov/electricity/monthly/",
        "source_version": "2024",
    },
    "cloud_egress_pricing": {
        "source_id": "Public-cloud WAN egress pricing direction",
    },
    "transport_estimate": {
        "source_id": "Edge/backhaul transport pricing (scenario estimate)",
    },
    "compute_pricing": {
        "source_id": "Cloud/edge compute pricing direction (per vCPU-second)",
    },
    "pa_efficiency": {
        "source_id": "Mobile PA efficiency (research range)",
    },
    "rf_frontend": {
        "source_id": "IoT/handset RF front-end circuit power (research estimate)",
    },
    "access_radio": {
        "source_id": "Ground MEC/BS access radio electronics (small-cell class, research estimate)",
    },
    "uav_radio": {
        "source_id": "UAV-BS access/backhaul radio electronics (research estimate)",
    },
    "dc_server_power": {
        "source_id": "Edge/metro/cloud server power (research estimate)",
    },
    "fibre_energy": {
        "source_id": "Wired transport energy-per-bit (research range)",
    },
    "eval_methodology": {
        "source_id": "Simulation evaluation methodology (common practice; no versioned standard)",
    },
    "example_route": {
        "source_id": "SmartCityEdgeSim example route",
        "source_version": "1.2",
    },
    "epa_egrid_2023": {
        "source_id": "US EPA eGRID national average CO2 emission rate",
        "source_uri": "https://www.epa.gov/egrid",
        "source_version": "2023",
    },
    "gsma_mobile_net_zero_2026": {
        "source_id": "GSMA Mobile Net Zero: State of the Industry on Climate Action",
        "source_uri": "https://www.gsma.com/newsroom/press-release/gsma-mobile-net-zero-report-nearly-a-quarter-of-all-electricity-purchased-by-global-mobile-industry-is-now-renewable/",
        "source_version": "2026 report, 2024 data",
    },
    "iea_data_centres_2024": {
        "source_id": "IEA, Energy demand from AI -- data centre renewable electricity share",
        "source_uri": "https://www.iea.org/reports/energy-and-ai/energy-demand-from-ai",
        "source_version": "2024",
    },
    "server_pcf_2024": {
        "source_id": "Dell R740 product carbon footprint, cross-validated by Boavizta component model",
        "source_uri": "https://boavizta.org/en/blog/empreinte-de-la-fabrication-d-un-serveur",
        "source_version": "2024",
    },
    "ran_pcf_ericsson": {
        "source_id": "Ericsson RBS 6201 radio base station product carbon footprint",
        "source_uri": "https://www.ericsson.com/en/blog/2021/5/how-to-estimate-carbon-emissions-from-mobile-networks",
    },
    "device_pcf_2025": {
        "source_id": "Smartphone-class product environmental reports (Pixel 10 / iPhone 17)",
        "source_uri": "https://www.techinsights.com/blog/hidden-cost-pro-your-smartphones-carbon-footprint-bigger-you-think",
        "source_version": "2025",
    },
    "drone_lca_review": {
        "source_id": "UAV/drone lifecycle carbon footprint review",
        "source_uri": "https://www.mdpi.com/2504-477X/9/4/169",
    },
    "hardware_refresh_cycle": {
        "source_id": "Server/RAN hardware refresh-cycle and embodied-carbon amortization practice",
    },
}


def cite(ref_id, *, maturity, validity_notes="", valid_from=None, valid_to=None):
    """Build a §11 provenance record from a registered source id.

    The source name/URL/version come from `_REFERENCES[ref_id]`; the caller supplies
    the value-specific `validity_notes` and the `maturity` claim. Output is the same
    provenance dict shape the runtime consumes — only the citation text is centralized.
    """
    if ref_id not in _REFERENCES:
        raise KeyError(f"Unknown reference {ref_id!r}. Add it to config/_references.py.")
    ref = _REFERENCES[ref_id]
    return provenance(
        ref["source_id"],
        source_uri=ref.get("source_uri", ""),
        source_version=ref.get("source_version", ""),
        valid_from=valid_from,
        valid_to=valid_to,
        validity_notes=validity_notes,
        maturity=maturity,
    )
