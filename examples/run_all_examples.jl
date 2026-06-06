using PrettyTables
using Printf

function run_all_julia_files(start_dir::String;
    backend::Symbol = :text,
    show_table::Bool = true,
    gc_before_each::Bool = false,
)
    results = NamedTuple{(:file, :path, :time_s, :alloc_bytes, :success), Tuple{String, String, Float64, Int, Bool}}[]


    safe_include(path::AbstractString) = try
        include(path)
        (true, nothing)
    catch e
        (false, e)
    end

    for (root, dirs, files) in walkdir(start_dir)
        sort!(dirs)
        sort!(files)

        for file in files
            endswith(file, ".jl") || continue
            file_path = joinpath(root, file)

            println("Running: $file_path")
            gc_before_each && GC.gc()

            stats = Base.@timed safe_include(file_path)

            val = hasproperty(stats, :value) ? stats.value : stats[1]
            success, err = val

            if success
                @info "🎉 Success: $file_path"
            else
                @info "💀 Error in $file_path"
                showerror(stdout, err)
                println()
            end

            push!(results, (
                file = file,
                path = file_path,
                time_s = stats.time,
                alloc_bytes = stats.bytes,
                success = success,
            ))
            println()
        end
    end

    if show_table
        n  = length(results)
        ok = count(r -> r.success, results)

        data = Matrix{Any}(undef, n, 4)
        for (i, r) in enumerate(results)
            data[i, 1] = r.file
            data[i, 2] = @sprintf("%.3f", r.time_s)
            data[i, 3] = @sprintf("%.3f", r.alloc_bytes / (1024.0^2))
            data[i, 4] = r.success ? "Yes" : "No"
        end

        pretty_table(
            data;
            column_labels = ["File", "Time (s)", "Alloc (MiB)", "OK?"],
            alignment     = [:l, :r, :r, :c],
            title         = "Julia file run summary ($ok/$n succeeded)",
            backend       = backend,
        )
    end

    return results
end
