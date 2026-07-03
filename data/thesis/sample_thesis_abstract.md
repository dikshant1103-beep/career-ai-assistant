# Thesis — sample abstract

**Title**: Hybrid physics-ML simulation of two-wheeler dynamics and energy
storage, with deployment on an interactive chassis workbench.

**Abstract**:
This thesis develops a unified stack for motorcycle chassis design that
couples (a) high-fidelity multibody-dynamics simulation, (b) physics-informed
neural networks for residual learning, and (c) an interactive workbench
exposing the entire pipeline to engineers. We contribute: a 16-state RK4
integrator in C++ with LibTorch PINN residuals, validated against four
analytic limits; a generalized-alpha implementation for stiff MBD problems
covering Phases 1–3 with 32/32 passing tests; a Pacejka MF tire stack with
Mz, combined slip, and contact-patch migration; and a 22-tab desktop
workbench achieving 44/44 (100%) validation. We also present MambaRUL, a
selective-state-space approach to battery remaining-useful-life prediction
that reduces MAE by ~14% versus an LSTM baseline.
