# Project: Volcanic heat melting an ice cap (GPGN 436/536 final project)
Goal: a Python app that solves a 1D heat-conduction Stefan problem (ice melting from a hot base) with the enthalpy method on a fixed grid, with a Panel user interface.

Rules:
- Always run Python and pip as .venv/bin/python and .venv/bin/python -m pip. The shell's default python is Anaconda and must not be used. Never run conda commands.
- Keep simulation code in model/solver.py with NO imports from panel, holoviews, or any UI library. It must run on its own from a script or notebook.
- Keep all interface code in app.py. It only imports and calls model/solver.py.
- Use SI units everywhere. Every plot axis and slider needs a physical unit label.
- verification.ipynb must import model/solver.py. Never write a second solver for verification.
- Comment the solver heavily. Each major block should say which term of the PDE it implements.
- Commit after each completed step with a clear message.
- Prefer simple, readable code over clever code. The student must be able to explain every line.
