using Documenter
using PEPit

const IS_CI = get(ENV, "CI", "false") == "true"
const DOCS_ROOT = @__DIR__
const PACKAGE_ROOT = normpath(joinpath(DOCS_ROOT, ".."))
const EXAMPLES_ROOT = joinpath(PACKAGE_ROOT, "examples")
const GENERATED_EXAMPLES_ROOT = joinpath(DOCS_ROOT, "src", "generated_examples")
const REPO_EXAMPLES_URL = "https://github.com/PerformanceEstimation/PEPit.jl/blob/main/examples"

function _title_from_slug(slug::AbstractString)
    return join(uppercasefirst.(split(replace(slug, "_" => " "))), " ")
end

function _extract_example_doc(path::AbstractString)
    text = read(path, String)
    m = match(r"(?s)@doc\s+raw\"\"\"(.*?)\"\"\"\s*function\s+(wc_[A-Za-z0-9_!]+)\s*\(", text)
    m === nothing && return nothing
    doc = replace(m.captures[1], r"^\n+" => "")
    doc = replace(doc, r"\s+$" => "")
    return (name = m.captures[2], doc = doc)
end

function _demote_markdown_headings(text::AbstractString; levels::Integer=1)
    return replace(text, r"(?m)^#{1,4}\s+" => heading -> begin
        first_non_hash = findfirst(!=('#'), heading)
        level = first_non_hash === nothing ? length(heading) : first_non_hash - 1
        repeat("#", level + levels) * heading[(level + 1):end]
    end)
end

function _prepare_example_doc(text::AbstractString)
    return _demote_markdown_headings(text; levels=1)
end

function _generate_example_doc_pages!()
    isdir(GENERATED_EXAMPLES_ROOT) && rm(GENERATED_EXAMPLES_ROOT; recursive=true)
    mkpath(GENERATED_EXAMPLES_ROOT)

    repo_tree_base = replace(REPO_EXAMPLES_URL, "/blob/" => "/tree/")

    # The Examples overview page is generated from the same scan as the per-example
    # pages, so every family is listed uniformly and links never drift.
    overview = IOBuffer()
    println(overview, "# Examples")
    println(overview)
    println(overview, "The Julia examples are ordinary scripts under ",
            "[`examples/`]($(repo_tree_base)), grouped by family. Each page below is ",
            "generated from the corresponding `wc_*` function docstring and includes the ",
            "problem statement, the algorithm, the performance metric, the reference ",
            "guarantee when available, the Julia arguments and return values, and a link ",
            "to the source file.")
    println(overview)

    pages = Pair{String,Any}[]
    for family in sort(filter(isdir, readdir(EXAMPLES_ROOT; join=true)))
        family_slug = basename(family)
        family_title = _title_from_slug(family_slug)
        entries = []

        for path in sort(filter(p -> endswith(p, ".jl"), readdir(family; join=true)))
            extracted = _extract_example_doc(path)
            extracted === nothing && continue
            example_slug = splitext(basename(path))[1]
            example_title = _title_from_slug(example_slug)
            rel_source = replace(relpath(path, EXAMPLES_ROOT), "\\" => "/")
            push!(entries, (title=example_title,
                            slug=example_slug,
                            source_url="$(REPO_EXAMPLES_URL)/$(rel_source)",
                            doc=extracted.doc))
        end

        isempty(entries) && continue

        family_output_root = joinpath(GENERATED_EXAMPLES_ROOT, family_slug)
        mkpath(family_output_root)

        family_pages = Pair{String,String}["Overview" => "generated_examples/$(family_slug)/index.md"]
        open(joinpath(family_output_root, "index.md"), "w") do io
            println(io, "# $(family_title)")
            println(io)
            println(io, "Examples in this category.")
            println(io)
            for entry in entries
                println(io, "- [$(entry.title)]($(entry.slug).md)")
            end
        end

        println(overview, "## $(family_title)")
        println(overview)
        println(overview, "[Source directory]($(repo_tree_base)/$(family_slug))")
        println(overview)

        for entry in entries
            output_path = joinpath(family_output_root, "$(entry.slug).md")
            open(output_path, "w") do io
                println(io, "# $(entry.title)")
                println(io)
                println(io, "[Source file]($(entry.source_url))")
                println(io)
                println(io, _prepare_example_doc(entry.doc))
                println(io)
            end
            push!(family_pages, entry.title => "generated_examples/$(family_slug)/$(entry.slug).md")
            println(overview, "- [$(entry.title)](generated_examples/$(family_slug)/$(entry.slug).md)")
        end
        println(overview)

        push!(pages, family_title => family_pages)
    end

    write(joinpath(DOCS_ROOT, "src", "examples.md"), String(take!(overview)))

    return pages
end

const EXAMPLE_DOC_PAGES = _generate_example_doc_pages!()

# Mirror the root changelog into the docs as the "Release notes" page.
cp(joinpath(PACKAGE_ROOT, "CHANGELOG.md"), joinpath(DOCS_ROOT, "src", "changelog.md"); force=true)

makedocs(;
    modules = [PEPit],
    sitename = "PEPit.jl",
    authors = "PEPit.jl contributors",
    remotes = nothing,
    linkcheck = false,
    checkdocs = :exports,
    warnonly = [:missing_docs],
    pagesonly = true,
    meta = Dict(:CurrentModule => PEPit),
    format = Documenter.HTML(;
        prettyurls = IS_CI,
        edit_link = "main",
        repolink = "https://github.com/PerformanceEstimation/PEPit.jl",
        assets = ["assets/pepit.css"],
    ),
    pages = [
        "Home" => "index.md",
        "Quick start" => "quickstart.md",
        "API reference" => [
            "Core workflow" => "api/core.md",
            "Function classes" => "api/functions.md",
            "Operator classes" => "api/operators.md",
            "Primitive steps" => "api/steps.md",
            "Utilities" => "api/utilities.md",
        ],
        "Examples" => vcat(["Overview" => "examples.md"], EXAMPLE_DOC_PAGES),
        "Tutorials" => "tutorials.md",
        "Contributing" => "contributing.md",
        "Release notes" => "changelog.md",
    ],
)

deploydocs(;
    repo = "github.com/PerformanceEstimation/PEPit.jl.git",
    devbranch = "main",
)
