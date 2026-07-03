# Sample Resume

**Mechanical / AI Engineer** | Battery systems, vehicle dynamics, FEM/CAD, ML

you@example.com | linkedin.com/in/you | github.com/you

---

## Summary
Engineering student building physics-informed ML systems for energy storage and vehicle dynamics. Hands-on with PyChrono, PINNs, Pacejka tire models, FEM, and CAD. Comfortable across C++ and Python research stacks.

## Education
- B.Tech, Mechanical Engineering — (in progress)
- Coursework: FEM, vehicle dynamics, control systems, ML, numerical methods

## Skills
- **Programming**: Python, C++, MATLAB, LaTeX, Bash
- **ML/AI**: PyTorch, scikit-learn, PINNs, Mamba state-space models, RAG, LangChain
- **Simulation**: PyChrono, MBD, Generalized-alpha solver, FEM
- **CAD/FEM**: SolidWorks, ANSYS, NX
- **Vehicle dynamics**: Pacejka MF tire model, motorcycle dynamics, suspension
- **Battery**: PyBaMM, RUL prediction, electrochemical modelling

## Experience / Projects

### Chassis Workbench (Lead, 22-tab Python/PyQt app)
- Built a 22-tab desktop workbench for motorcycle chassis design covering geometry, suspension, MBD, validation.
- Implemented gear-limited V_max calculation and AS% + R header for steering geometry.
- Generated 8-slide validation pack achieving 44/44 (100%) verification tests.

### MBD Motorcycle Simulator (PyChrono, C++/RK4 + LibTorch PINN)
- Wrote a 16-state RK4 integrator in C++ with LibTorch-backed PINN residuals.
- Fixed four critical bugs (front-wheel coupling, yaw sign, gyroscopic term, stability) verified against analytic limits.
- Phases 1-3 of MBD roadmap complete: 32/32 unit tests passing.

### Tire Rig Simulation
- Implemented Pacejka Magic Formula incl. Mz, combined slip, contact-patch migration.
- WebSocket server + Electron dashboard for real-time visualisation.
- Next step: wire relaxation ODE with measured kappa from wheel spin.

### Battery RUL — MambaRUL
- Adapted Mamba state-space architecture for battery remaining-useful-life prediction.
- Trained on PyBaMM-augmented dataset; submitted to IEEE conference.

## Publications
- "MambaRUL: A State-Space Approach to Battery RUL Estimation" — IEEE submission, 2026.

## Languages
- English (fluent), Hindi (native)
