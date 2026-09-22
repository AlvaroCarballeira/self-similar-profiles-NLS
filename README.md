# Computer-assisted proof code: Self-similar profiles for energy-supercritical, defocusing NLS.

This code accompanies the article [Blow-up for energy supercritical defocusing nonlinear Schrödinger equations in dimensions three, four and five](https://arxiv.org/abs/2609.23685v1).

## Authors

- Kevin Buck (Brown University) <kevin_buck@brown.edu>
- Alvaro Carballeira (Brown University), <alvaro_carballeira_mora@brown.edu>
- Javier Gómez-Serrano (Brown University) <javier_gomez_serrano@brown.edu>
- Jia Shi (Indiana University Bloomington) <js289@iu.edu>

## Organization

For \((d,p)=(3,27),(4,9),(5,7)\), it rigorously certifies the existence of a smooth, global, nonvanishing radial self-similar profile that is regular at the origin and has a nonzero pure slow tail at infinity (equivalently, \(C_f=0\) and \(C_s\ne0\)). `prove.py` reads an approximate guess from the corresponding `candidate-d*-p*.json`, checks the origin and exterior hypotheses for the fixed point argument, propagates their interval enclosures, and verifies the Brouwer conditions.

This code follows the manuscript's mathematical argument: 

- `helpers.py` holds the three supported pairs, their run defaults, and shared Arb and certificate utilities.
- `origin.py` constructs the regular-origin polynomial and checks hypothesis for nearby exact solutions by a contraction.
- `exterior.py` constructs the pure-slow asymptotic expansions, bounds the linear modes and full residual, and checks existence of nearby exact solutions by a contraction.
- `propagation.py` bounds every Taylor polynomial's ODE errors and transports 
solutions and their variations with respect to (A, \Omega, K) to the matching radius.
- `matching.py` forms the matching map \(\mathcal H\), the finite and exact matching errors, and checks the Brouwer bounds \(Y,E,Z\).
- `prove.py` chooses the dimension case and parameter cube, runs all the proof stages, and saves the proof's closing bounds.

Install `requirements.txt` and run `prove.py` for each dimension:

```bash
python prove.py --d 3
python prove.py --d 4
python prove.py --d 5
```

Rigorous results are saved in this directory as `certificate-d*-p*.json`.

The numerical search in `generate_candidate.py` is independent of the proof. It is used to generate numerical candidates around which the proof is closed.
It generates `candidate-d*-p*.json` files. To construct these candidates:

```bash
python generate_candidate.py --d 3
python generate_candidate.py --d 4
python generate_candidate.py --d 5
```

## Runtimes

The table below reports wall-clock times on a single machine: Intel Core i7-12700H (14 cores, 20 hardware threads), 16 GB RAM, Linux, with Python 3.12.3, python-flint 0.9.0, mpmath 1.4.1, NumPy 2.5.3, and SciPy 1.18.1.

Script | Case | Output | Wall-clock
--- | --- | --- | ---
`prove.py` | `d = 3`, `p = 27` | `certificate-d3-p27.json` | 8 min 32 s
`prove.py` | `d = 4`, `p = 9` | `certificate-d4-p9.json` | 7 min 34 s
`prove.py` | `d = 5`, `p = 7` | `certificate-d5-p7.json` | 5 min 46 s
`generate_candidate.py` | `d = 3`, `p = 27` | `candidate-d3-p27.json` | 1 h 10 min 53 s
`generate_candidate.py` | `d = 4`, `p = 9` | `candidate-d4-p9.json` | 42 min 21 s
`generate_candidate.py` | `d = 5`, `p = 7` | `candidate-d5-p7.json` | 38 min 21 s


## License

This software is available under the [MIT License](LICENSE).
