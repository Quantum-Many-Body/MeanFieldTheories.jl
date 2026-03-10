using Documenter
using MeanFieldTheories

makedocs(;
    modules=[MeanFieldTheories],
    authors="Yong-Yue Zong",
    repo="https://github.com/Quantum-Many-Body/MeanFieldTheories.jl/blob/{commit}{path}#{line}",
    sitename="MeanFieldTheories.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Quantum-Many-Body.github.io/MeanFieldTheories.jl",
        repolink="https://github.com/Quantum-Many-Body/MeanFieldTheories.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Quantum System" => "quantumsystem.md",
        "Hartree-Fock Approximation" => [
            "Real Space"      => "hartreefock_real.md",
            "Momentum Space"  => "hartreefock_momentum.md",
        ],
        "Single-Mode Approximation" => "singlemode.md",
        "Examples" => [
            "SDW-CDW Phase Diagram" => "SDW_CDW.md",
        ],
        "API Reference" => "api.md",
    ],
    checkdocs=:none,
    warnonly=true,
)

deploydocs(;
    repo="github.com/Quantum-Many-Body/MeanFieldTheories.jl",
    devbranch="main",
    push_preview=true,
)
