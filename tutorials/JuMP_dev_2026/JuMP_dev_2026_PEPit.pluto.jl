### A Pluto.jl notebook ###
# v0.20.27

using Markdown
using InteractiveUtils

# ╔═╡ e74b449d-f45f-4752-a0bb-c3f174a0ac9e
# ╠═╡ pluto_cell_id = "e74b449d-f45f-4752-a0bb-c3f174a0ac9e"
begin
    import Pkg;
    Pkg.activate();
    using PEPit;
    using OrderedCollections;
    using Clarabel;
    using JuMP;
    using Plots;
    using LinearAlgebra;
    using PlutoUI;
end

# ╔═╡ 1022ae02-d98b-4841-8f40-e2cd0038951f
# ╠═╡ pluto_cell_id = "1022ae02-d98b-4841-8f40-e2cd0038951f"
md"""
# `PEPit.jl`: computer-assisted worst-case analysis of first-order optimization algorithms in Julia

$(HTML("<div style='text-align:center; font-size:1.45rem; font-style:italic; margin-top:0.4rem;'>Shuvomoy Das Gupta</div>"))

$(HTML("<div style='text-align:center; font-size:1.45rem; font-style:italic; margin-top:0.4rem;'>Joint work with Python PEPit team: B. Goujaud, C. Moucer, F. Glineur, J. Hendrickx, A. Taylor, A. Dieuleveut.</div>"))


---

$(LocalResource(
    joinpath(@__DIR__, "JuMP_dev_logo_2026.png"),
    :style => "display:block; width:260px; max-width:35%; height:auto; margin:1.0rem auto 0.6rem auto;",
))

$(HTML("<div style='text-align:center; font-size:1.45rem; font-style:italic; margin-top:0.4rem;'>JuMP-dev 2026, Edinburgh</div>"))

---
"""

# ╔═╡ 63f052b9-71dc-46c9-a133-8e164d86d64d
# ╠═╡ pluto_cell_id = "63f052b9-71dc-46c9-a133-8e164d86d64d"
md"""
## Some personal update

Since the last time we met at JuMP-dev 2023:

* I finished my PhD in 2024.
* I was a postdoc at Columbia IEOR during 2024-2025.
* In Fall 2025, I started a faculty position at Rice University in Houston, TX.

"""

# ╔═╡ d10681f1-6f10-4d84-9dcb-2233f3ba26a0
# ╠═╡ pluto_cell_id = "d10681f1-6f10-4d84-9dcb-2233f3ba26a0"
md"""
## What is `PEPit.jl`?

* `PEPit.jl` is a native Julia implementation of the Performance Estimation Programming (PEP) methodology [Drori--Teboulle 2014; Taylor--Hendrickx--Glineur 2017a,b].

* Functionally, `PEPit.jl` is equivalent to the Python package `PEPit` [Goujaud et al. 2024]. 

* **The core idea in PEP.** Model the analysis of first-order optimization algorithms as higher-level optimization problems called performance estimation problems (PEPs).

    * These PEPs are convex semidefinite programs (SDPs)!

    * We solve these SDPs numerically to obtain tight worst-case bounds for known algorithms. 

    * We can also use PEP to discover new algorithms under suitable conditions!

"""

# ╔═╡ 8ac339ad-7344-4f59-b5d3-ce75b337d578
# ╠═╡ pluto_cell_id = "8ac339ad-7344-4f59-b5d3-ce75b337d578"
md"""
## Why `PEPit.jl`?

* The `Julia`+`JuMP` ecosystem is very conducive to open-source reproducible research.
* `JuMP` supports 50+ commercial and open-source solvers.
  * PEP can benefit from interaction between different types of solvers.
* `Julia` is a fast language, `JuMP` provides solver-independent callbacks (lazy constraints, user-cuts, heuristic solutions).
  * PEP often requires exploiting problem-specific structures to extract useful information.
  * `Julia`+`JuMP` can allow us to write custom code to exploit structures present in PEPs.
* `JuMP` already has many useful packages, such as `Dualization.jl`, `ParametricOptInterface.jl`, and `BilevelJuMP.jl`, that can potentially exploit problem-specific structure in PEPs.
* Certain types of PEPs can be naturally nonlinear/nonconvex: `Julia`+`JuMP` can be a natural fit for these types of PEPs.

"""

# ╔═╡ 97a3ba18-9aae-499c-a2f4-5ea61cd74684
# ╠═╡ pluto_cell_id = "97a3ba18-9aae-499c-a2f4-5ea61cd74684"
md"""
## PEP workflow 

A performance estimation problem (PEP) turns the worst-case convergence bound of an optimization algorithm into a convex optimization problem.

**Core PEP workflow**

1. **Input.** Specify the mathematical setting (input to `PEPit.jl`):
   - function class, e.g. ``f\in\mathcal{F}_{\mu,L}``,
   - algorithm, e.g. ``x_{i+1}=x_{i}-\alpha \nabla f(x_i)``,
   - performance metric, e.g. ``\|x_{i}-x_\star\|^2``,
   - initialization, e.g. ``\|x_0-x_\star\|^2\leq 1``.

2. **Sample.** Sample only the objects experienced by the method:
   - points ``x_i``,
   - gradients ``g_i``,
   - function values ``f_i``.

3. **Interpolation.** Replace the unknown infinite-dimensional function/operator by *interpolation* constraints.

4. **Gramian transformation.** Lift inner products and norm-squared terms into a Gram matrix: ``G_{ij}=\langle v_i,v_j\rangle`` and ``G\succeq 0``. Function values are embedded in vector ``F=[\{f_i\}_i]``.

6. **Solve.** Solve the resulting SDP!
"""

# ╔═╡ feba6fe7-952b-4a24-9d8a-b9ae89db574e
# ╠═╡ pluto_cell_id = "feba6fe7-952b-4a24-9d8a-b9ae89db574e"
md"""
## A simple example 

* **Optimization problem.** Suppose we are interested in solving ``\min_{x} f(x)``, where ``f:\mathbb{R}^d \to \mathbb{R}``.

* **Function type.** We consider ``f \in \mathcal{F}_{\mu,L}``.

  * ``\mathcal{F}_{\mu,L}``: the class of ``L``-smooth and ``\mu``-strongly convex functions (functions that are not too flat or sharp)

* **Algorithm to be studied.** The algorithm is gradient descent:

```math
x_{i+1} = x_{i} - \alpha \nabla f(x_i),\; i=0,1,2,\ldots,\quad \text{(GD)}
```

* We start with some ``x_0`` and then update the iterates using (GD).

"""

# ╔═╡ 142a83d4-f305-4aee-8d3a-a5876e9239b7
# ╠═╡ pluto_cell_id = "142a83d4-f305-4aee-8d3a-a5876e9239b7"
md"""
## Main question

* **Question. [Taylor 2024]** What is the smallest ``\tau`` such that 

```math
\frac{\|x_{i+1}-x_\star\|^2}{\|x_i-x_\star\|^2}
\leq
\tau 
```

for every dimension ``d``, every ``f\in\mathcal{F}_{\mu,L}``, every starting point ``x_0``, and the gradient step ``x_{i+1}=x_i-\alpha\nabla f(x_i)``?

* Note that, without loss of generality, we can set ``i \gets 0``.
"""

# ╔═╡ 30e916f1-4d6e-4446-9d03-e9cbf31ce6c2
# ╠═╡ pluto_cell_id = "30e916f1-4d6e-4446-9d03-e9cbf31ce6c2"
md"""
## Abstract optimization problem

* Find the smallest ``\tau`` such that

```math
\|x_1-x_\star\|^2 \leq \tau \|x_0-x_\star\|^2
```

for every dimension ``d\in\mathbb{N}``, every ``L``-smooth and ``\mu``-strongly convex function ``f\in\mathcal{F}_{\mu,L}``, every starting point ``x_0``, the algorithmic step ``x_1=x_0-\alpha\nabla f(x_0)``, and ``x_\star\in\arg\min_x f(x)``.


* The problem can be written as the *abstract* optimization problem:

```math
\begin{aligned}
\tau =
\text{maximize}_{f,x_0,x_1,x_\star,d}\quad
& \frac{\|x_1-x_\star\|^2}{\|x_0-x_\star\|^2}\\
\text{subject to}\quad
& f\in\mathcal{F}_{\mu,L},\\
& x_1=x_0-\alpha\nabla f(x_0),\\
& \nabla f(x_\star)=0.
\end{aligned}
```

Variables: ``f,x_0,x_1,x_\star,d``. Parameters: ``\mu,L,\alpha``.

"""

# ╔═╡ a4525d8a-fe32-445c-89ef-f60e3b0384ce
# ╠═╡ pluto_cell_id = "a4525d8a-fe32-445c-89ef-f60e3b0384ce"
md"""
## Sampled version

* **Abstract problem.** The original PEP has an infinite-dimensional function as a variable:

```math
\begin{aligned}
\text{maximize}_{f,x_0,x_1,x_\star,d}\quad
& \frac{\|x_1-x_\star\|^2}{\|x_0-x_\star\|^2}\\
\text{subject to}\quad
& f\text{ is }L\text{-smooth and }\mu\text{-strongly convex},\\
& x_1=x_0-\alpha\nabla f(x_0),\\
& \nabla f(x_\star)=0.
\end{aligned}
```

* **Sampled problem.** The sampled version replaces ``f`` by finitely many unknown samples at ``x_0`` and ``x_\star``:

```math
\begin{aligned}
\text{maximize}_{\substack{x_0,x_1,x_\star\\ g_0,g_\star\\ f_0,f_\star,d}}\quad
& \frac{\|x_1-x_\star\|^2}{\|x_0-x_\star\|^2}\\
\text{subject to}\quad
& \exists f\in\mathcal{F}_{\mu,L}\text{ such that }
\begin{cases}
f_i=f(x_i), & i=0,\star,\\
g_i=\nabla f(x_i), & i=0,\star,
\end{cases}\\
& x_1=x_0-\alpha g_0,\\
& g_\star=0.
\end{aligned}
```

New variables: ``x_0,x_1,x_\star,g_0,g_\star,f_0,f_\star,d``.
"""

# ╔═╡ 7062a2b0-d5d2-406a-917b-55cc74ab5f18
# ╠═╡ pluto_cell_id = "7062a2b0-d5d2-406a-917b-55cc74ab5f18"
md"""
## Concept of interpolation

* **Setup.** Let ``I`` be an index set and suppose we are given samples

```math
\{(x_i,g_i,f_i)\}_{i\in I},
```

where ``x_i`` are points, ``g_i`` are gradients, and ``f_i`` are function values.

* **Interpolation question.**

```math
\text{does there exist }f\in\mathcal{F}_{\mu,L}
\text{ such that }
f(x_i)=f_i,\quad \nabla f(x_i)=g_i
\quad \forall i\in I?
```

* **Interpolation condition.** A necessary and sufficient condition is that, for all ``i,j\in I``,

```math
f_i \geq f_j+\langle g_j,x_i-x_j\rangle
+\frac{1}{2L}\|g_i-g_j\|^2
+\frac{\mu}{2(1-\mu/L)}
\left\|x_i-x_j-\frac{1}{L}(g_i-g_j)\right\|^2.
```

"""

# ╔═╡ cb464efd-f977-4034-867a-ef6407edab5e
# ╠═╡ pluto_cell_id = "cb464efd-f977-4034-867a-ef6407edab5e"
md"""
## Existence constraint ``\leftrightarrow`` interpolation constraint

* The sampled PEP contains the existence constraint

```math
\exists f\in\mathcal{F}_{\mu,L}\text{ such that }
\begin{cases}
f_i=f(x_i), & i=0,\star,\\
g_i=\nabla f(x_i), & i=0,\star.
\end{cases}
```

* Interpolation allows us to replace the existence constraint by explicit inequalities between the two sampled points:

```math
\begin{aligned}
f_\star &\geq f_0+\langle g_0,x_\star-x_0\rangle
+\frac{1}{2L}\|g_\star-g_0\|^2
+\frac{\mu}{2(1-\mu/L)}
\left\|x_\star-x_0-\frac{1}{L}(g_\star-g_0)\right\|^2,\\
f_0 &\geq f_\star+\langle g_\star,x_0-x_\star\rangle
+\frac{1}{2L}\|g_0-g_\star\|^2
+\frac{\mu}{2(1-\mu/L)}
\left\|x_0-x_\star-\frac{1}{L}(g_0-g_\star)\right\|^2.
\end{aligned}
```

* With ``x_1=x_0-\alpha g_0`` and ``g_\star=0``, this has the same optimal value as the sampled PEP: no relaxation has happened yet.

* What changed is the form of the problem: it is now a finite but *nonconvex* QCQP.
"""

# ╔═╡ 7bfe3022-5fe5-4234-9263-450c2eddee37
# ╠═╡ pluto_cell_id = "7bfe3022-5fe5-4234-9263-450c2eddee37"
md"""
## Semidefinite lifting

* **Key observation.** The problem now contains only scalar function values and inner products/norm squares involving the iterates and gradients in ``\mathbb{R}^d``. 

* **Gram matrix.** Define

```math
P \triangleq [x_0-x_\star,\ g_0]\in\mathbb{R}^{d\times 2},
\qquad
F\triangleq f_0-f_\star.
```

* The Gram matrix

```math
G\triangleq P^T P =
\begin{bmatrix}
\|x_0-x_\star\|^2 & \langle g_0,x_0-x_\star\rangle\\
\langle g_0,x_0-x_\star\rangle & \|g_0\|^2
\end{bmatrix}
\succeq 0, \quad \textup{rank} G \leq d
```

linearizes all squared norms and inner products! 


* **Large-scale assumption.** For ``d \geq 2``, ``\textup{rank} G \leq 2``, so the rank constraint is satisfied automatically. 

* **Final SDP.** Substituting ``x_1`` and ``g_\star``, and normalizing ``G_{1,1}=1`` gives the ``2\times 2`` SDP.

```math
\begin{aligned}
\text{maximize}_{G,F}\quad
& G_{1,1}+\alpha^2G_{2,2}-2\alpha G_{1,2}\\
\text{subject to}\quad
& F+\frac{L\mu}{2(L-\mu)}G_{1,1}
+\frac{1}{2(L-\mu)}G_{2,2}
-\frac{L}{L-\mu}G_{1,2}\leq 0,\\
& -F+\frac{L\mu}{2(L-\mu)}G_{1,1}
+\frac{1}{2(L-\mu)}G_{2,2}
-\frac{\mu}{L-\mu}G_{1,2}\leq 0,\\
& G_{1,1}=1,\qquad G\succeq 0.
\end{aligned}
```
"""

# ╔═╡ 1e7ed284-5e1b-42bf-8402-208572e16b4c
# ╠═╡ pluto_cell_id = "1e7ed284-5e1b-42bf-8402-208572e16b4c"
md"""
## PEP reformulation recap

* The workflow is consistent across other setups as well.

* Computation of the worst-case convergence bound is represented by:

  * vector ``F`` containing function values ``f_i``,
  * a Gram matrix ``G`` encoding inner products between points and gradients,
  * linear constraints encoding interpolation and initialization,
  * a linear objective encoding the performance metric.

* The formulated SDP will be of the form:

```math
\begin{aligned}
\text{maximize}_{G,F}\quad & \text{linear function of }(F, G) \\
\text{subject to}\quad & G\succeq 0,\\
& \text{linear interpolation constraints in } (F, G),\\
& \text{linear initial conditions in } (F, G).
\end{aligned}
```

* `PEPit.jl` lets us write the PEP in its most natural mathematical description and does the conversion to SDP automatically. 
"""

# ╔═╡ c696078d-8d7a-4288-a410-b47dd8b28fe9
# ╠═╡ pluto_cell_id = "c696078d-8d7a-4288-a410-b47dd8b28fe9"
md"""
## Solving the example problem in `PEPit.jl`
"""

# ╔═╡ 52ce76ce-b8be-4a74-9fce-0a79bb8dbc45
# ╠═╡ pluto_cell_id = "52ce76ce-b8be-4a74-9fce-0a79bb8dbc45"
md"""
**Initialize PEP.**
Let us start with defining an empty PEP, which we will construct step by step. 
"""

# ╔═╡ 8fd2eabd-2930-45f0-b97d-31f45fb5c61b
# ╠═╡ pluto_cell_id = "8fd2eabd-2930-45f0-b97d-31f45fb5c61b"
problem = PEP()

# ╔═╡ 6d026776-f591-4e2d-ad10-7437afea2666
# ╠═╡ pluto_cell_id = "6d026776-f591-4e2d-ad10-7437afea2666"
md"""
**Define function parameters.**
Define the function parameters ``\mu,L,\alpha``.
"""

# ╔═╡ 49648dd5-9d8c-4c19-b2c5-8077c55b8cd3
# ╠═╡ pluto_cell_id = "49648dd5-9d8c-4c19-b2c5-8077c55b8cd3"
begin
	mu=0.1
	L=1.0
	alpha=1.0
end

# ╔═╡ 554fe1cd-c189-455f-affc-712235bd2f03
# ╠═╡ pluto_cell_id = "554fe1cd-c189-455f-affc-712235bd2f03"
md"""
**Define function class**
Now we define the function class ``\mathcal{F}_{\mu, L}`` itself. Here `reuse_gradient=true` means that our function is differentiable. 
"""

# ╔═╡ a4272299-8adc-418b-bfb5-2089da17ee56
# ╠═╡ pluto_cell_id = "a4272299-8adc-418b-bfb5-2089da17ee56"
func = declare_function!(problem, SmoothStronglyConvexFunction, OrderedDict("mu" => mu, "L" => L); reuse_gradient=true)

# ╔═╡ c3ba8753-ed91-48d9-a7af-1d071e172372
# ╠═╡ pluto_cell_id = "c3ba8753-ed91-48d9-a7af-1d071e172372"
md"""
**Define initial+optimal points.**
Define the initial point ``x_0`` as `x0` and the optimal point ``x_\star`` as `xs`.
"""

# ╔═╡ 841888dd-0f1a-44f0-a7d2-dd2b0a69e67a
# ╠═╡ pluto_cell_id = "841888dd-0f1a-44f0-a7d2-dd2b0a69e67a"
xs = stationary_point!(func)

# ╔═╡ d8b1f447-ac40-47b4-8b57-24bb4db57baa
# ╠═╡ pluto_cell_id = "d8b1f447-ac40-47b4-8b57-24bb4db57baa"
x0 = set_initial_point!(problem)

# ╔═╡ 17ec4504-ac14-442b-93e2-0c6fee933f46
# ╠═╡ pluto_cell_id = "17ec4504-ac14-442b-93e2-0c6fee933f46"
md"""
**Define algorithm.**
Now define the gradient descent step:
```math
x_1 = x_0 - \alpha \nabla f(x_0)
```
"""

# ╔═╡ 3223cba6-b1a3-4d59-877e-bd7f78f219ce
# ╠═╡ pluto_cell_id = "3223cba6-b1a3-4d59-877e-bd7f78f219ce"
x1 = x0 - alpha * gradient!(func, x0)

# ╔═╡ 8d831c47-dd5e-4167-8284-878b8412504a
# ╠═╡ pluto_cell_id = "8d831c47-dd5e-4167-8284-878b8412504a"
md"""
**Initial condition.**
Next, define the *initial condition*: ``\|x_0-x_\star\|^2\leq 1``.
"""

# ╔═╡ 964c357a-5b8e-4fea-b00d-a6eeaf924565
# ╠═╡ pluto_cell_id = "964c357a-5b8e-4fea-b00d-a6eeaf924565"
set_initial_condition!(problem, (x0 - xs)^2 <= 1)

# ╔═╡ f7167d84-ae31-4e5e-9d23-f6915b581053
# ╠═╡ pluto_cell_id = "f7167d84-ae31-4e5e-9d23-f6915b581053"
md"""
**Performance metric.**
Finally, define the *performance metric*, which measures how far we are from the optimal point ``x_\star``. Smaller values are better; zero means the iterate is optimal.
"""

# ╔═╡ 21b0c6d3-a35d-4663-b99c-82c282dd159c
# ╠═╡ pluto_cell_id = "21b0c6d3-a35d-4663-b99c-82c282dd159c"
set_performance_metric!(problem, (x1 - xs)^2)

# ╔═╡ e65f5e53-ae24-48f5-b22d-e460efe60075
# ╠═╡ pluto_cell_id = "e65f5e53-ae24-48f5-b22d-e460efe60075"
md"""
## Solving the PEP
We now have everything needed to build the PEP. It is time to solve it.
"""

# ╔═╡ c81ac352-207d-4ae7-9753-3d4f9f30c134
# ╠═╡ pluto_cell_id = "c81ac352-207d-4ae7-9753-3d4f9f30c134"
primal_result = solve!(problem; verbose = true, return_full_model = true)

# ╔═╡ 34a73c98-50ab-478c-a218-eb7b318947aa
# ╠═╡ pluto_cell_id = "34a73c98-50ab-478c-a218-eb7b318947aa"
tau_star = primal_result.wc_value

# ╔═╡ ecdd5c0c-878c-4145-a987-309181cbea05
# ╠═╡ pluto_cell_id = "ecdd5c0c-878c-4145-a987-309181cbea05"
G_star = JuMP.value.(primal_result.variables.G)

# ╔═╡ 11f17139-c1a5-4d82-85bc-9563cd701bd7
# ╠═╡ pluto_cell_id = "11f17139-c1a5-4d82-85bc-9563cd701bd7"
F_star = JuMP.value.(primal_result.variables.F)

# ╔═╡ 82a4d004-2d3c-4ec5-ae8a-1f98f04d5a40
# ╠═╡ pluto_cell_id = "82a4d004-2d3c-4ec5-ae8a-1f98f04d5a40"
md"""
## Parameter sweeps for the gradient-step PEP

We wrap the one-step gradient descent PEP in a function that takes ``\mu`` and ``\alpha`` as inputs.

Throughout these sweeps, we fix ``L=1`` and compute the worst-case contraction factor ``\tau^\star(\mu,\alpha)``.
"""

# ╔═╡ dd070a70-c323-4770-afdc-aeb7cd9fb43f
# ╠═╡ pluto_cell_id = "dd070a70-c323-4770-afdc-aeb7cd9fb43f"
function gd_contraction_pep_problem(; gd_mu::Real, gd_alpha::Real, gd_L::Real=1.0)
    gd_problem = PEP()

    gd_func = declare_function!(
        gd_problem,
        SmoothStronglyConvexFunction,
        OrderedDict("mu" => Float64(gd_mu), "L" => Float64(gd_L));
        reuse_gradient=true,
    )

    gd_xs = stationary_point!(gd_func)
    gd_x0 = set_initial_point!(gd_problem)
    gd_x1 = gd_x0 - Float64(gd_alpha) * gradient!(gd_func, gd_x0)

    set_initial_condition!(gd_problem, (gd_x0 - gd_xs)^2 <= 1)
    set_performance_metric!(gd_problem, (gd_x1 - gd_xs)^2)

    return gd_problem
end

# ╔═╡ 4d3436d2-a476-4d34-872c-e6ef7c44ba20
# ╠═╡ pluto_cell_id = "4d3436d2-a476-4d34-872c-e6ef7c44ba20"
function gd_worst_case_tau(;
    gd_mu::Real,
    gd_alpha::Real,
    gd_L::Real=1.0,
    gd_solver=Clarabel.Optimizer,
    gd_verbose::Bool=false,
)
    gd_problem = gd_contraction_pep_problem(
        gd_mu=gd_mu,
        gd_alpha=gd_alpha,
        gd_L=gd_L,
    )

    gd_result = solve!(
        gd_problem;
        solver=gd_solver,
        verbose=gd_verbose,
        return_full_model=true,
    )

    return gd_result.wc_value
end

# ╔═╡ 456ed996-c2df-4a69-b00d-0f24b17bc0ff
# ╠═╡ pluto_cell_id = "456ed996-c2df-4a69-b00d-0f24b17bc0ff"
gd_theory_tau(; gd_mu::Real, gd_alpha::Real, gd_L::Real=1.0) =
    max(abs(1 - gd_alpha * gd_L), abs(1 - gd_alpha * gd_mu))^2

# ╔═╡ 4339f34c-8b04-44ad-b351-cf2b85ba85fa
# ╠═╡ pluto_cell_id = "4339f34c-8b04-44ad-b351-cf2b85ba85fa"
md"""
## Plot of ``\tau`` vs ``\mu``

Fix ``L=1`` and ``\alpha=1``. Vary ``\mu`` from ``0.1`` to ``0.85``.
"""

# ╔═╡ 4e56113f-26fa-4600-a542-9bc2bb11aae9
# ╠═╡ pluto_cell_id = "4e56113f-26fa-4600-a542-9bc2bb11aae9"
begin
    gd_L_for_mu_sweep = 1.0
    gd_alpha_for_mu_sweep = 1.0
    gd_mu_grid = unique(sort(vcat(collect(0.1:0.1:0.8), 0.85)))

    gd_tau_vs_mu = [
        gd_worst_case_tau(
            gd_mu=gd_mu,
            gd_alpha=gd_alpha_for_mu_sweep,
            gd_L=gd_L_for_mu_sweep,
        )
        for gd_mu in gd_mu_grid
    ]

    gd_tau_theory_vs_mu = [
        gd_theory_tau(
            gd_mu=gd_mu,
            gd_alpha=gd_alpha_for_mu_sweep,
            gd_L=gd_L_for_mu_sweep,
        )
        for gd_mu in gd_mu_grid
    ]
end

# ╔═╡ 6167708a-ff66-431f-a64b-56e17e9ae2bc
# ╠═╡ pluto_cell_id = "6167708a-ff66-431f-a64b-56e17e9ae2bc"
begin
    plot(
        gd_mu_grid,
        gd_tau_vs_mu;
        label="PEPit.jl SDP",
        marker=:circle,
        linewidth=3,
        xlabel="μ",
        ylabel="τ",
        title="Worst-case τ versus μ for L=1, α=1",
        legend=:topright,
        grid=true,
    );
    
    plot!(
        gd_mu_grid,
        gd_tau_theory_vs_mu;
        label="theory",
        linestyle=:dash,
        linewidth=3,
    )
end

# ╔═╡ 7e2d6adf-f42e-4828-b436-13188015abaa
# ╠═╡ pluto_cell_id = "7e2d6adf-f42e-4828-b436-13188015abaa"
begin
    gd_L_for_alpha_sweep = 1.0
    gd_mu_for_alpha_sweep = 0.1
    gd_alpha_grid = collect(0.1:0.1:2.0)

    gd_tau_vs_alpha = [
        gd_worst_case_tau(
            gd_mu=gd_mu_for_alpha_sweep,
            gd_alpha=gd_alpha,
            gd_L=gd_L_for_alpha_sweep,
        )
        for gd_alpha in gd_alpha_grid
    ]

    gd_tau_theory_vs_alpha = [
        gd_theory_tau(
            gd_mu=gd_mu_for_alpha_sweep,
            gd_alpha=gd_alpha,
            gd_L=gd_L_for_alpha_sweep,
        )
        for gd_alpha in gd_alpha_grid
    ]
end

# ╔═╡ 7fe56322-4b75-480a-a225-98284139d961
# ╠═╡ pluto_cell_id = "7fe56322-4b75-480a-a225-98284139d961"
md"""
## Plot of ``\tau`` vs ``\alpha``

Fix ``L=1`` and ``\mu=0.1``. Vary ``\alpha`` from ``0.1`` to ``2.0`` with granularity ``0.1``.
"""

# ╔═╡ de29c13e-071c-406a-8c13-d67b111125d0
# ╠═╡ pluto_cell_id = "de29c13e-071c-406a-8c13-d67b111125d0"
begin
    plot(
        gd_alpha_grid,
        gd_tau_vs_alpha;
        label="PEPit.jl SDP",
        marker=:circle,
        linewidth=3,
        xlabel="α",
        ylabel="τ",
        title="Worst-case τ versus α for fixed μ=0.1, L=1",
        legend=:topright,
        grid=true,
    )
    plot!(
        gd_alpha_grid,
        gd_tau_theory_vs_alpha;
        label="theory",
        linestyle=:dash,
        linewidth=3,
    )
end

# ╔═╡ c3f53877-8785-4764-9d8c-a2d13b2e262e
# ╠═╡ pluto_cell_id = "c3f53877-8785-4764-9d8c-a2d13b2e262e"
md"""
## Internal architecture of `PEPit.jl`

* At a high level, it follows the architecture of Python `PEPit`.

* The Julia package uses `struct`s, `methods`, and `multiple dispatch` instead of Python-style classes.

The package is built around a small set of mathematical objects:

| What? | Why? |
| --- | --- |
| `Point` | vector-like quantity: iterate, gradient, residual |
| `Expression` | scalar quantity: function value, norm square, inner product |
| `Constraint` | scalar equality or inequality |
| `PSDMatrix` | linear matrix inequality |
| `PEPFunction` | sampled function with interpolation constraints |
| `PEP` | container for assumptions, initial conditions, metrics, and SDP assembly |
"""

# ╔═╡ 39701215-dc94-49fd-a792-ca8f667f7ccd
# ╠═╡ pluto_cell_id = "39701215-dc94-49fd-a792-ca8f667f7ccd"
md"""
## Step 1: user input becomes a `PEP`

The user describes a worst-case analysis problem through four ingredients:

* **Problem class:** assumptions such as ``f\in\mathcal{F}_{\mu,L}``.
* **Algorithm:** a first-order recurrence generating ``x_1,\ldots,x_N``.
* **Initial condition:** a normalization such as ``\mathcal{I}(x_0,\ldots)\leq 1``.
* **Performance measure:** the quantity ``\mathcal{P}(x_N,\ldots)`` to maximize.

In `PEPit.jl`, these ingredients are stored in a `PEP` object.

Mathematically, the `PEP` represents

```math
\tau^\star
=
\sup
\left\{
\mathcal{P}(x_N,\ldots,x_0,f)
:
f\in\mathcal{F},\;
\mathcal{I}(x_0,\ldots,f)\leq 1,\;
\text{algorithmic recurrence holds}
\right\}.
```

So `PEP` is the container that accumulates:

* sampled functions,
* symbolic points and expressions,
* initial conditions,
* performance metrics,
* constraints needed to assemble the final SDP.

"""

# ╔═╡ a10d813c-5433-47da-a811-52a52a0d3b62
# ╠═╡ pluto_cell_id = "a10d813c-5433-47da-a811-52a52a0d3b62"
md"""
## Step 2: symbolic execution creates `Point` and `Expression` objects

`PEPit.jl` first runs the algorithm symbolically.

* Vector-like quantities are represented by `Point` objects:

```math
x_0,\; x_1,\ldots,x_N,\qquad
g_i=f^\prime(x_i),\qquad
r_i,\ldots
```

* Scalar quantities are represented by `Expression` objects:

```math
f_i=f(x_i),\qquad
\|x_i-x_j\|^2,\qquad
\langle g_i,x_j-x_k\rangle.
```

* For example, a gradient step

```math
x_{i+1}=x_i-\alpha_i f^\prime(x_i)
```

is stored as an algebraic relation between `Point` objects.

* The performance measure and initial condition are `Expression` inequalities:

```math
\mathcal{P}(x_N,\ldots) \quad \text{and} \quad
\mathcal{I}(x_0,\ldots)\leq 1.
```


"""

# ╔═╡ cb614bcc-b97e-481f-bb37-0b80489d3275
# ╠═╡ pluto_cell_id = "cb614bcc-b97e-481f-bb37-0b80489d3275"
md"""
## Step 3: `PEPFunction` generates interpolation `Constraint`s

* The problem class is represented by a `PEPFunction`.

* For each sampled point ``x_i``, the `PEPFunction` introduces symbolic data

```math
(x_i,g_i,f_i),
\qquad
g_i=f^\prime(x_i),
\qquad
f_i=f(x_i).
```

* The infinite-dimensional statement

```math
f\in\mathcal{F}
```

is replaced by finitely many interpolation `Constraint`s:

```math
\{(x_i,g_i,f_i)\}_{i\in I}
\text{ is extendable to some } f\in\mathcal{F}.
```

* For ``\mathcal{F}_{\mu,L}``, these are exactly the smooth strongly convex interpolation inequalities seen earlier.

Thus:

* `PEPFunction` knows the function class,
* it samples gradients and function values,
* it contributes the interpolation `Constraint`s to the `PEP`.
"""

# ╔═╡ 1733a775-720e-4267-81ef-835d35379ff3
# ╠═╡ pluto_cell_id = "1733a775-720e-4267-81ef-835d35379ff3"
md"""
## Step 4: `Constraint`s are linearized through the Gram matrix and `PSDMatrix`

* All vector quantities appearing in the PEP are collected into a list

```math
p_1,\ldots,p_m.
```

* These may include iterates, gradients, residuals, or differences of such objects.

`PEPit.jl` introduces the Gram matrix

```math
G =
\begin{bmatrix}
\langle p_1,p_1\rangle & \cdots & \langle p_1,p_m\rangle\\
\vdots & \ddots & \vdots\\
\langle p_m,p_1\rangle & \cdots & \langle p_m,p_m\rangle
\end{bmatrix}
\succeq 0.
```

* Every norm square and inner product becomes linear in ``G``:

```math
\|p_i\|^2 = G_{i,i},
\qquad
\langle p_i,p_j\rangle = G_{i,j}.
```

* The condition ``G\succeq 0`` is represented internally as a `PSDMatrix`.

So the conversion is:

```math
\text{Point geometry}
\quad\longrightarrow\quad
\text{Gram matrix }G
\quad\longrightarrow\quad
\texttt{PSDMatrix}.
```
"""

# ╔═╡ a6aefe0a-f317-4834-b5ee-682a3a26b09d
# ╠═╡ pluto_cell_id = "a6aefe0a-f317-4834-b5ee-682a3a26b09d"
md"""
## Step 5: the `PEP` assembles the final SDP

After the Gram lifting, the `PEP` has everything needed to build the SDP.

* `Point` objects determine the Gram matrix entries.
* `Expression` objects become linear functions of ``G`` and scalar values ``F``.
* `Constraint` objects become linear equalities or inequalities.
* `PSDMatrix` objects become semidefinite constraints.
* `PEPFunction` objects contribute interpolation constraints.
* The `PEP` object assembles all pieces into a JuMP model.

The final SDP has the form

```math
\begin{aligned}
\text{maximize}_{G,F}\quad
& \langle C,G\rangle + c^\top F\\
\text{subject to}\quad
& \langle A_\ell,G\rangle + a_\ell^\top F \leq b_\ell,
\qquad \ell=1,\ldots,m,\\
& G\succeq 0.
\end{aligned}
```

The primal SDP returns the worst-case value ``\tau^\star``.  
The dual SDP gives a proof certificate:

```math
\mathcal{P}(x_N,\ldots,x_0,f)
\leq
\tau^\star
\mathcal{I}(x_0,\ldots,f).
```

In short:

```math
\texttt{PEP}
\Rightarrow
\texttt{Point}+\texttt{Expression}+\texttt{PEPFunction}
\Rightarrow
\texttt{Constraint}+\texttt{PSDMatrix}
\Rightarrow
\text{JuMP}.
```
"""

# ╔═╡ ccf14962-1f16-47ce-a483-60cb09ec1b2e
# ╠═╡ pluto_cell_id = "ccf14962-1f16-47ce-a483-60cb09ec1b2e"
md"""
## Function classes and primitive steps

The package includes core function classes such as:

- convex functions,
- smooth functions,
- strongly convex functions,
- smooth strongly convex functions,
- convex Lipschitz functions,
- convex indicators.

It also includes primitive algorithmic steps:

- gradient and inexact gradient steps,
- proximal and inexact proximal steps,
- Bregman gradient/proximal steps,
- exact line search,
- linear and shifted optimization steps.

These are small compositional blocks for building many PEPs without rewriting interpolation logic.
"""

# ╔═╡ b0a00f04-dfd4-421e-bb66-a74f1c79b43c
# ╠═╡ pluto_cell_id = "b0a00f04-dfd4-421e-bb66-a74f1c79b43c"
md"""
## Operator splitting

Beyond smooth optimization, `PEPit.jl` can model operator-splitting problems, such as 

- monotone operators,
- nonexpansive operators,
- Lipschitz operators,
- linear operators.

This supports examples such as:

- accelerated proximal point,
- Douglas-Rachford splitting,
- three-operator splitting,
- fixed-point iterations.

The same workflow applies: define objects, run the recurrence, set the metric, and solve the PEP.
"""

# ╔═╡ 557095a3-0008-4692-9e38-67b3cbe32f08
# ╠═╡ pluto_cell_id = "557095a3-0008-4692-9e38-67b3cbe32f08"
md"""
## Different optimization setups: stochastic, online, potential, and worst-case examples

The example suite also includes:

- stochastic gradient descent,
- online gradient descent,
- nonconvex gradient descent,
- optimized gradient methods,
- potential-function examples, 
- low-dimensional worst-case scenario searches.

For a JuMP audience, this is the important extensibility point:

```math
\text{new modeling ingredient}
\quad\Rightarrow\quad
\text{new interpolation/constraint block}
\quad\Rightarrow\quad
\text{same JuMP solve path}.
```

* Note that there is a dedicated package called `AutoLyap` with both Python and Julia implementations for potential function analysis [Upadhyaya et al. 2026].

"""

# ╔═╡ fb2c9d4d-1947-4ed9-babe-ba848965f5af
# ╠═╡ pluto_cell_id = "fb2c9d4d-1947-4ed9-babe-ba848965f5af"
md"""
## Future plans

We are working toward incorporating step-size optimization in `PEPit.jl` and Python `PEPit`:

```math
\begin{aligned}
\text{inner problem:}\quad
& \text{certify a worst-case bound for fixed algorithm parameters},\\
\text{outer problem:}\quad
& \text{optimize over step sizes or algorithm coefficients}.
\end{aligned}
```

Here:

- the inner dual provides a certificate structure (`PEPit.jl` can already solve it separately),
- the outer problem searches over algorithm parameters,
- branch-and-bound [Das Gupta--Van Parys--Ryu 2024] or other nonlinear methods [Kamri--Hendrickx--Glineur 2025] can handle the nonconvexity,
- in some cases [Drori--Taylor 2020; Taylor--Drori 2023], step-size optimization can also be solved as a convex SDP,
- the output should be both a method and a proof.

This is the bridge from analyzing a method to discovering new, more efficient methods!
"""

# ╔═╡ 426504e7-6c24-4c22-bf44-0ba4dac836c6
# ╠═╡ pluto_cell_id = "426504e7-6c24-4c22-bf44-0ba4dac836c6"
md"""
## Thank you!
"""

# ╔═╡ 095fb967-7e76-4367-beb9-4b2fe70900d6
# ╠═╡ pluto_cell_id = "095fb967-7e76-4367-beb9-4b2fe70900d6"
md"""
## References

[Drori--Teboulle 2014] Y. Drori and M. Teboulle. *Performance of first-order methods for smooth convex minimization: a novel approach.* Mathematical Programming, 2014.

[Taylor--Hendrickx--Glineur 2017a] A. B. Taylor, J. M. Hendrickx, and François Glineur. *Smooth strongly convex interpolation and exact worst-case performance of first-order methods.* Mathematical Programming, 2017.

[Taylor--Hendrickx--Glineur 2017b] A. B. Taylor, J. M. Hendrickx, and François Glineur. *Exact worst-case performance of first-order methods for composite convex optimization.* SIAM Journal on Optimization, 2017.

[Goujaud et al. 2024] B. Goujaud, C. Moucer, François Glineur, J. M. Hendrickx, A. B. Taylor, and A. Dieuleveut. *PEPit: computer-assisted worst-case analyses of first-order optimization methods in Python.* Mathematical Programming Computation, 2024.

[Taylor 2024] A. B. Taylor. *Towards principled and systematic approaches to the analysis and design of optimization algorithms.* 2024.

[Das Gupta--Van Parys--Ryu 2024] S. Das Gupta, B. P. G. Van Parys, and E. K. Ryu. *Branch-and-bound performance estimation programming: a unified methodology for constructing optimal optimization methods.* Mathematical Programming, 2024. 

[Kamri--Hendrickx--Glineur 2025] Y. Kamri, J. M. Hendrickx, and François Glineur. *Numerical design of optimized first-order algorithms.* arXiv:2507.20773, 2025.

[Upadhyaya et al. 2026] M. Upadhyaya, S. Das Gupta, A. B. Taylor, S. Banert, and P. Giselsson. *The AutoLyap software suite for computer-assisted Lyapunov analyses of first-order methods.* arXiv:2506.24076, 2026.

[Drori--Taylor 2020] Y. Drori and A. B. Taylor. *Efficient first-order methods for convex minimization: a constructive approach.* Mathematical Programming, 2020. 

[Taylor--Drori 2023] A. B. Taylor and Y. Drori. *An optimal gradient method for smooth strongly convex minimization.* Mathematical Programming, 2023.
"""

# ╔═╡ Cell order:
# ╟─e74b449d-f45f-4752-a0bb-c3f174a0ac9e
# ╟─1022ae02-d98b-4841-8f40-e2cd0038951f
# ╟─63f052b9-71dc-46c9-a133-8e164d86d64d
# ╟─d10681f1-6f10-4d84-9dcb-2233f3ba26a0
# ╟─8ac339ad-7344-4f59-b5d3-ce75b337d578
# ╟─97a3ba18-9aae-499c-a2f4-5ea61cd74684
# ╟─feba6fe7-952b-4a24-9d8a-b9ae89db574e
# ╟─142a83d4-f305-4aee-8d3a-a5876e9239b7
# ╟─30e916f1-4d6e-4446-9d03-e9cbf31ce6c2
# ╟─a4525d8a-fe32-445c-89ef-f60e3b0384ce
# ╟─7062a2b0-d5d2-406a-917b-55cc74ab5f18
# ╟─cb464efd-f977-4034-867a-ef6407edab5e
# ╟─7bfe3022-5fe5-4234-9263-450c2eddee37
# ╟─1e7ed284-5e1b-42bf-8402-208572e16b4c
# ╟─c696078d-8d7a-4288-a410-b47dd8b28fe9
# ╟─52ce76ce-b8be-4a74-9fce-0a79bb8dbc45
# ╠═8fd2eabd-2930-45f0-b97d-31f45fb5c61b
# ╟─6d026776-f591-4e2d-ad10-7437afea2666
# ╠═49648dd5-9d8c-4c19-b2c5-8077c55b8cd3
# ╟─554fe1cd-c189-455f-affc-712235bd2f03
# ╠═a4272299-8adc-418b-bfb5-2089da17ee56
# ╟─c3ba8753-ed91-48d9-a7af-1d071e172372
# ╠═841888dd-0f1a-44f0-a7d2-dd2b0a69e67a
# ╠═d8b1f447-ac40-47b4-8b57-24bb4db57baa
# ╟─17ec4504-ac14-442b-93e2-0c6fee933f46
# ╠═3223cba6-b1a3-4d59-877e-bd7f78f219ce
# ╟─8d831c47-dd5e-4167-8284-878b8412504a
# ╠═964c357a-5b8e-4fea-b00d-a6eeaf924565
# ╟─f7167d84-ae31-4e5e-9d23-f6915b581053
# ╠═21b0c6d3-a35d-4663-b99c-82c282dd159c
# ╟─e65f5e53-ae24-48f5-b22d-e460efe60075
# ╠═c81ac352-207d-4ae7-9753-3d4f9f30c134
# ╠═34a73c98-50ab-478c-a218-eb7b318947aa
# ╠═ecdd5c0c-878c-4145-a987-309181cbea05
# ╠═11f17139-c1a5-4d82-85bc-9563cd701bd7
# ╟─82a4d004-2d3c-4ec5-ae8a-1f98f04d5a40
# ╟─dd070a70-c323-4770-afdc-aeb7cd9fb43f
# ╟─4d3436d2-a476-4d34-872c-e6ef7c44ba20
# ╟─456ed996-c2df-4a69-b00d-0f24b17bc0ff
# ╟─4339f34c-8b04-44ad-b351-cf2b85ba85fa
# ╟─4e56113f-26fa-4600-a542-9bc2bb11aae9
# ╟─6167708a-ff66-431f-a64b-56e17e9ae2bc
# ╟─7e2d6adf-f42e-4828-b436-13188015abaa
# ╟─7fe56322-4b75-480a-a225-98284139d961
# ╟─de29c13e-071c-406a-8c13-d67b111125d0
# ╟─c3f53877-8785-4764-9d8c-a2d13b2e262e
# ╟─39701215-dc94-49fd-a792-ca8f667f7ccd
# ╟─a10d813c-5433-47da-a811-52a52a0d3b62
# ╟─cb614bcc-b97e-481f-bb37-0b80489d3275
# ╟─1733a775-720e-4267-81ef-835d35379ff3
# ╟─a6aefe0a-f317-4834-b5ee-682a3a26b09d
# ╟─ccf14962-1f16-47ce-a483-60cb09ec1b2e
# ╟─b0a00f04-dfd4-421e-bb66-a74f1c79b43c
# ╟─557095a3-0008-4692-9e38-67b3cbe32f08
# ╟─fb2c9d4d-1947-4ed9-babe-ba848965f5af
# ╟─426504e7-6c24-4c22-bf44-0ba4dac836c6
# ╟─095fb967-7e76-4367-beb9-4b2fe70900d6
