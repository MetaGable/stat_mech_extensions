# 2D damped wave equation (reflective box)

This is a real-time 2D visual simulation of the damped, driven wave equation (matching the structure in your screenshot, ignoring subscripts/superscripts):

\[
\left[\frac{1}{\gamma^2}\frac{\partial^2}{\partial t^2} + \frac{2}{\gamma}\frac{\partial}{\partial t} + 1 - r^2\nabla^2\right]\phi(\mathbf{r},t)=Q(\mathbf{r},t)
\]

In code we evolve the equivalent second-order form:

\[
\phi_{tt} = c^2\nabla^2\phi - 2\gamma\,\phi_t - \omega_0^2\,\phi + S
\]

with parameters exposed at the top of `wave_sim.py`. (This is the same “damped wave + restoring term + source” structure; the constant mapping is noted in the file.)

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python wave_sim.py
```

## Controls

- **Space**: pause/unpause
- **R**: reset field to 0
- **[ / ]**: decrease / increase damping `gamma`
- **- / =**: decrease / increase source amplitude
- **, / .**: decrease / increase source frequency
- **Esc**: quit

The source `Q` is applied on the **bottom edge** over the **center 1/6 segment** of the box and oscillates sinusoidally.

