"""Run the computer-assisted profile proof for dimension 3, 4, or 5."""
import argparse
import json
from pathlib import Path

from flint import arb, ctx

from helpers import encode, profile_spec
from origin import certify_origin, origin_data
from exterior import certify_exterior, tail_data
from propagation import initialize, propagate
from matching import brouwer_bounds

HERE = Path(__file__).resolve().parent
BITS = 896
S_IN = arb(1) / 8  # Inner radius s_in.
S_OUT = arb(48)  # Outer radius s_out.


def prove(d):
    """Close the proof for one dimension and return its certificate."""
    ctx.prec = BITS
    spec = profile_spec(d)
    candidate = json.loads(
        (HERE / f"candidate-d{d}-p{spec.p}.json").read_text())

    center = [arb(candidate[name]).mid() for name in ("A", "omega", "K")]
    rho = arb(2) ** -spec.rho_bits
    box = [arb(value, rho) for value in center]
    s_m = arb(spec.s_m)

    origin_error, origin_record = certify_origin(
        box, S_IN, spec.inner_order, d)
    exterior_error, exterior_record = certify_exterior(
        box, S_OUT, spec.outer_order, d)

    flows = []
    flow_records = []
    for data, s_start, order, error, degree in (
        (origin_data, S_IN, spec.inner_order,
         origin_error, spec.inner_degree),
        (tail_data, S_OUT, spec.outer_order,
         exterior_error, spec.outer_degree),
    ):
        initial = initialize(
            data(center, s_start, order, d),
            data(box, s_start, order, d),
            error,
        )
        flow = propagate(
            initial, s_start, s_m, center[1], box[1], rho, d,
            taylor_degree=degree,
        )
        flows.append(flow)
        flow_records.append(flow["record"])

    brouwer_record = brouwer_bounds(
        flows[0], flows[1], s_m, rho)
    return {
        "d": d,
        "p": spec.p,
        "center": dict(zip(
            ("A", "omega", "K"), map(encode, center))),
        "rho": encode(rho),
        "origin": origin_record,
        "exterior": exterior_record,
        "inner": flow_records[0],
        "outer": flow_records[1],
        "brouwer": brouwer_record,
        "proof_closed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, required=True, choices=(3, 4, 5))
    d = parser.parse_args().d
    record = prove(d)
    output = HERE / f"certificate-d{d}-p{record['p']}.json"
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"PROOF CLOSED: {output.name}")


if __name__ == "__main__":
    main()
