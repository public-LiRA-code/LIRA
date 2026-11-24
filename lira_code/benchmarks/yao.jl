
using Yao, CUDA, JSON, Logging

disable_logging(Logging.Warn)

function build_circuit(case, ::Type{T}) where {T}
    n = case["n"]
    blocks = AbstractBlock[]
    slots = Int[]
    rot1 = Dict("rx" => Rx, "ry" => Ry, "rz" => Rz)
    fixed1 = Dict("h" => H, "x" => X, "y" => Y, "z" => Z, "s" => ConstGate.S, "t" => ConstGate.T)
    for g in case["gates"]
        name = g["name"]
        q = Int.(g["q"]) .+ 1
        if haskey(rot1, name)
            push!(blocks, put(n, q[1] => rot1[name](T(g["theta"]))))
            push!(slots, g["p"] + 1)
        elseif haskey(rot1, name[2:end]) && name[1] == 'f'
            push!(blocks, put(n, q[1] => matblock(Matrix{Complex{T}}(mat(rot1[name[2:end]](T(g["theta"])))))))
        elseif name == "p"
            push!(blocks, put(n, q[1] => shift(T(g["theta"]))))
            push!(slots, g["p"] + 1)
        elseif name == "swap"
            push!(blocks, swap(n, q[1], q[2]))
        elseif name == "rzz"
            push!(blocks, put(n, (q[1], q[2]) => rot(kron(Z, Z), T(g["theta"]))))
            push!(slots, g["p"] + 1)
        elseif haskey(fixed1, name)
            push!(blocks, put(n, q[1] => fixed1[name]))
        elseif name == "cx" || name == "cnot"
            push!(blocks, control(n, q[1], q[2] => X))
        elseif name == "cz"
            push!(blocks, control(n, q[1], q[2] => Z))
        else
            error("gate not handled: $name")
        end
    end
    return chain(n, blocks), slots
end

function hamiltonian(case, ::Type{T}) where {T}
    n = case["n"]
    haskey(case, "terms") || return sum([put(n, q => Z) for q in 1:n]) / n
    pauli = Dict("X" => X, "Y" => Y, "Z" => Z)
    blocks = AbstractBlock[]
    for (c, ps) in case["terms"]
        isempty(ps) && continue
        push!(blocks, T(c) * kron(n, [parse(Int, q) + 1 => pauli[p] for (q, p) in ps]...))
    end
    return sum(blocks)
end

mutable struct PeakMem
    proc
    path::String
end

function start_peak()
    path = tempname()
    proc = run(pipeline(`nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits -lms 20`, stdout=path, stderr=devnull), wait=false)
    while !isfile(path) || isempty(strip(read(path, String)))
        sleep(0.01)
    end
    return PeakMem(proc, path)
end

function stop_peak(pm::PeakMem)
    sleep(0.05)
    kill(pm.proc)
    v = [parse(Float64, x) for x in split(read(pm.path, String)) if !isempty(x)]
    rm(pm.path, force=true)
    return (maximum(v) - v[1]) / 1024
end

function run_case(path, ::Type{T}, repeat, gpu) where {T}
    case = JSON.parsefile(path)
    n = case["n"]
    row = Dict{String,Any}("engine" => "yao", "case" => case["case"], "n" => n, "gates" => length(case["gates"]),
                           "params" => case["n_params"], "dtype" => (T == Float32 ? "c64" : "c128"),
                           "yao" => string(pkgversion(Yao)), "cuda_jl" => string(pkgversion(CUDA)), "gpu" => gpu)
    try
        circuit, slots = build_circuit(case, T)
        h = hamiltonian(case, T)
        constant = haskey(case, "terms") ? sum(Float64[c for (c, ps) in case["terms"] if isempty(ps)]; init=0.0) : 0.0
        newreg() = gpu ? cu(zero_state(Complex{T}, n)) : zero_state(Complex{T}, n)
        runs = Dict{String,Float64}[]
        cost = 0.0
        grad = zeros(Float64, case["n_params"])
        GC.gc(true)
        gpu && CUDA.reclaim()
        pm = gpu ? start_peak() : nothing
        for it in 1:repeat
            t0 = time()
            reg = newreg()
            c = gpu ? (CUDA.@sync real(expect(h, reg => circuit))) : real(expect(h, reg => circuit))
            t1 = time()
            reg = newreg()
            res = gpu ? (CUDA.@sync expect'(h, reg => circuit)) : expect'(h, reg => circuit)
            t2 = time()
            push!(runs, Dict("cost_only" => t1 - t0, "gradient" => t2 - t1))
            cost = Float64(real(c)) + constant
            g = Float64.(collect(res.second))
            fill!(grad, 0.0)
            for (k, s) in enumerate(slots)
                grad[s] += g[k]
            end
            reg = nothing
            res = nothing
            (t2 - t0) > 40 && break
        end
        gpu && (row["peak_gib"] = stop_peak(pm))
        best = runs[argmin([r["gradient"] for r in runs])]
        row["total"] = best["gradient"]
        row["cost_only"] = best["cost_only"]
        row["runs"] = runs
        row["cost"] = cost
        row["grad"] = grad
        println("yao ($(row["dtype"]), $(gpu ? "GPU" : "CPU"))  $(rpad(case["case"], 16)) n=$n  cost-only $(round(best["cost_only"], digits=4))s  " *
                "gradient (fwd+obs+bwd) $(round(best["gradient"], digits=4))s  cost=$(round(cost, digits=8))  " *
                "peak mem $(round(get(row, "peak_gib", 0.0), digits=2)) GiB")
    catch e
        row["error"] = first(sprint(showerror, e), 300)
        println("yao  $(rpad(case["case"], 16)) n=$n  FAILED ", row["error"])
        GC.gc(true)
        gpu && CUDA.reclaim()
    end
    flush(stdout)
    return row
end

function main()
    outdir = ARGS[1]
    T = ARGS[2] == "c128" ? Float64 : Float32
    repeat = parse(Int, ARGS[3])
    gpu = ARGS[4] != "cpu"
    mkpath(outdir)
    run_case(ARGS[5], T, 1, gpu)
    for path in ARGS[5:end]
        row = run_case(path, T, repeat, gpu)
        open(joinpath(outdir, "yao_$(row["case"])_n$(row["n"]).json"), "w") do f
            JSON.print(f, row)
        end
    end
end

main()
