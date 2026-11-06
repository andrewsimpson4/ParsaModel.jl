using UnicodePlots, Term, OrderedCollections

const _SUNSET_BAR = ["#FF5E62", "#FF7560", "#FF8C5E", "#FFA35C", "#FFBA60", "#FFD166"]
const _OCEAN_BAR  = ["#FF7E5F", "#E5748F", "#C471ED", "#7A8CE2", "#2CA5E0", "#00B4DB"]
const _PLOT_RGB_MAP = [
    "\e[38;5;8m" => "\e[38;2;255;166;71m",
    "\e[38;5;4m" => "\e[38;2;0;136;204m",
    "\e[38;5;5m" => "\e[38;2;56;189;248m",
    "\e[38;5;1m" => "\e[38;2;125;211;252m",
    "\e[38;5;3m" => "\e[38;2;255;255;255m",
    "\e[38;5;2m" => "\e[38;2;56;189;248m",
    "\e[38;5;6m" => "\e[38;2;186;230;253m",
    "\e[38;5;7m" => "\e[38;2;255;255;255m",
    "steps"      => "\e[38;2;255;166;71msteps\e[0m"
]

mutable struct EM_TUI
    n_obs::Int
    n_init::Int
    eps::Float64
    init_eps::Float64
    n_packages::Int
    build_stage::Int
    build_msg::String
    build_start_ns::UInt64
    build_elapsed::Float64
    em_start_ns::UInt64
    last_render_ns::UInt64
    total_iters::Int
    render_count::Int
    last_lines::Int
    init_warnings::OrderedDict{String, Int}
    final_warnings::OrderedDict{String, Int}
    prev_logger::Any
    verbose::Bool
    had_warning::Bool
    in_final::Bool
end

struct TUIWarningLogger <: Base.CoreLogging.AbstractLogger
    tui::EM_TUI
end

Base.CoreLogging.min_enabled_level(::TUIWarningLogger) = Base.CoreLogging.Warn
Base.CoreLogging.shouldlog(::TUIWarningLogger, level, _module, group, id) = level >= Base.CoreLogging.Warn
Base.CoreLogging.catch_exceptions(::TUIWarningLogger) = true

function Base.CoreLogging.handle_message(logger::TUIWarningLogger, level, message, _module, group, id, file, line; kwargs...)
    logger.tui.had_warning = true
    if logger.tui.verbose
        msg = strip(string(message))
        if !isempty(msg)
            target = logger.tui.in_final ? logger.tui.final_warnings : logger.tui.init_warnings
            target[msg] = get(target, msg, 0) + 1
        end
    elseif !isnothing(logger.tui.prev_logger)
        Base.CoreLogging.handle_message(logger.tui.prev_logger, level, message, _module, group, id, file, line; kwargs...)
    end
    return nothing
end

function EM_TUI(n_obs::Int, n_init::Int, eps::Real, init_eps::Real, verbose::Bool=true)
    now_ns = time_ns()
    tui = EM_TUI(n_obs, n_init, Float64(eps), Float64(init_eps), 0, 0, "Initializing...", now_ns, 0.0, now_ns, UInt64(0), 0, 0, 0, OrderedDict{String, Int}(), OrderedDict{String, Int}(), nothing, verbose, false, false)
    tui.prev_logger = Base.CoreLogging.global_logger(TUIWarningLogger(tui))
    return tui
end

function restore_logger!(tui::EM_TUI)
    if !isnothing(tui.prev_logger)
        Base.CoreLogging.global_logger(tui.prev_logger)
        tui.prev_logger = nothing
    end
end

function _tui_bar(frac::Real, width::Int=20; palette::Vector{String}=_SUNSET_BAR)
    f = clamp(Float64(frac), 0.0, 1.0)
    filled = Int(round(f * width))
    empty = width - filled
    buf = IOBuffer()
    n_col = length(palette)
    for i in 1:filled
        c_idx = clamp(Int(ceil((i / max(1, width)) * n_col)), 1, n_col)
        col = palette[c_idx]
        print(buf, "{$(col)}█{/$(col)}")
    end
    pct = lpad(string(Int(round(f * 100))), 3) * "%"
    print(buf, "{dim}", "░"^empty, "{/dim} ", pct)
    return String(take!(buf))
end

const _SURFER_PALETTE = Dict{Char, String}(
    'H' => "255;209;102", # Sunlit Gold hair/head
    'S' => "255;175;120", # Warm Sunlit skin
    'W' => "255;94;98",   # Sunset Coral wetsuit
    'B' => "255;209;102", # Sunlit Gold surfboard
    'F' => "255;255;255", # Pure White whitecap foam
    'I' => "202;240;248", # Seafoam Ice spray
    'C' => "72;202;228",  # Electric Cyan wave barrel wall
    'T' => "0;180;219",   # Ocean Turquoise wave face
    'D' => "0;119;182",   # Deep Ocean Blue swell
)

const _BRAILLE_DOTS = UInt8[
    0x01 0x08;
    0x02 0x10;
    0x04 0x20;
    0x40 0x80
]

# 24x28 sub-cell Braille bitmaps (6 char rows x 14 char cols):
# Rows 1-3: Surfer head, torso/arms, and bent knees
# Row 4:    Defined surfboard (dot rows 13-15 with upturned pointed nose & 1-dot gap above wave)
# Row 5:    Gleaming Pure White (#FFFFFF) & Seafoam Ice (#CAF0F8) foaming whitecap roller
# Row 6:    Ocean Turquoise (#00B4DB) & Deep Blue (#0077B6) wave swell base
const _STANCE_HIGH = [
    "..........HHHH............",
    ".........HHHHHHH..........",
    "...SS.....HHSSSS..........",
    "....SS.....SSSS...........",
    ".....SS..WWWWWWW...SSSS...",
    "......SWWWWWWWWWWSSSS.....",
    "........WWWWWWWWWW........",
    "........WWWWWWWWW.........",
    ".......WWWW....WWWW.......",
    "......WWWW......WWWW......",
    ".....SSSS........SSSS.....",
    "....SSS...........SSS.....",
    "....SSS...........SSS..BBB",
    "BBBBBBBBBBBBBBBBBBBBBBBBBB",
    ".BBBBBBBBBBBBBBBBBBBBBBB..",
    ".BB.......................",
    "FFFFFFFFFFFFFFIIIIIIIIFFFF",
    ".FFFFFFFFFFFFIIIIIIIIIIFFF",
    "..FFFFFFFFFFIIIIIIIIIIIIFF",
    "...FFFFFFFFIIIIIIIIIIIIFF.",
    "....CCCCCCCCTTTTTTTTTTTTCC",
    ".....TTTTTTTDDDDDDDDTTTTT.",
    ".......DDDDDDDDDDDDDDDD...",
    ".........DDDDDDDDDDDD....."
]

const _STANCE_CROUCH = [
    "..........................",
    "..........HHHH............",
    "..SS.....HHHHHHH..........",
    "...SS.....HHSSSS..........",
    "....SS....WWWWWW..........",
    ".....SS..WWWWWWWW..SSSS...",
    "......SWWWWWWWWWWSSSS.....",
    "........WWWWWWWWWW........",
    ".......WWWWWWWWWW.........",
    "......WWWW....WWWW........",
    ".....WWWW......WWWW.......",
    "....SSSS........SSSS......",
    "....SSS..........SSS...BBB",
    "BBBBBBBBBBBBBBBBBBBBBBBBBB",
    ".BBBBBBBBBBBBBBBBBBBBBBB..",
    ".BB.......................",
    ".FFFFFFFFFFFFFIIIIIIIIFFFF",
    "..FFFFFFFFFFFIIIIIIIIIIFFF",
    "...FFFFFFFFFIIIIIIIIIIIIFF",
    "....FFFFFFFIIIIIIIIIIIIIFF",
    ".....CCCCCCCTTTTTTTTTTTTCC",
    "......TTTTTTDDDDDDDDTTTTT.",
    "........DDDDDDDDDDDDDDD...",
    "..........DDDDDDDDDDD....."
]

function _render_shifted_stance(grid::Vector{String}, dx::Int)
    H = length(grid)
    W = length(grid[1])
    char_h = cld(H, 4)
    char_w = cld(W + 2, 2)
    cells = Matrix{Tuple{Char, String}}(undef, char_h, char_w)
    for cr in 1:char_h, cc in 1:char_w
        mask = UInt8(0)
        chosen_color = ""
        for dy in 1:4, sub_x in 1:2
            r = (cr - 1) * 4 + dy
            c = (cc - 1) * 2 + sub_x - dx
            if 1 <= r <= H && 1 <= c <= W
                ch = grid[r][c]
                if ch != '.'
                    mask |= _BRAILLE_DOTS[dy, sub_x]
                    chosen_color = _SURFER_PALETTE[ch]
                end
            end
        end
        bchar = Char(0x2800 + UInt32(mask))
        cells[cr, cc] = (bchar, chosen_color)
    end
    return cells
end

# Smooth 6-step sub-pixel surfing loop: small left-right (dx=0..2 dots) & up-down (high/crouch) motion
const _SURF_CARVE_FRAMES = [
    _render_shifted_stance(_STANCE_HIGH,   0),
    _render_shifted_stance(_STANCE_HIGH,   1),
    _render_shifted_stance(_STANCE_HIGH,   2),
    _render_shifted_stance(_STANCE_CROUCH, 2),
    _render_shifted_stance(_STANCE_CROUCH, 1),
    _render_shifted_stance(_STANCE_CROUCH, 0),
]

function _overlay_surfer_in_plot(str_plt::String, frame::Int)
    lines = split(str_plt, '\n')
    canvas_rows = Int[]
    for (idx, line) in enumerate(lines)
        if occursin('│', line) && !occursin('┌', line) && !occursin('└', line)
            push!(canvas_rows, idx)
        end
    end
    n_rows = length(canvas_rows)
    # Slow down frame cadence so the carving motion is gentle and realistic
    anim_tick = div(frame, 3)
    carve_idx = mod(anim_tick, length(_SURF_CARVE_FRAMES)) + 1
    cells = _SURF_CARVE_FRAMES[carve_idx]
    sprite_h, sprite_w = size(cells)
    if n_rows < sprite_h + 1
        return str_plt
    end

    first_row_chars = collect(lines[canvas_rows[1]])
    canvas_w = count(c -> ('⠀' <= c <= '⣿'), first_row_chars)
    if canvas_w < sprite_w + 2
        return str_plt
    end

    # Color the top crest of the log-likelihood wave in frothy whitecap / seafoam
    for crest_r in 1:min(2, n_rows)
        r_idx = canvas_rows[crest_r]
        r_chars = collect(lines[r_idx])
        r_braille = findall(c -> ('⠀' <= c <= '⣿'), r_chars)
        styled = String[string(c) for c in r_chars]
        col_str = crest_r == 1 ? "\e[38;2;255;255;255m" : "\e[38;2;224;251;252m"
        for c in 1:length(r_braille)
            idx_c = r_braille[c]
            ch = r_chars[idx_c]
            if ch != '⠀'
                styled[idx_c] = col_str * string(ch) * "\e[0m"
            end
        end
        lines[r_idx] = join(styled)
    end

    # Tuck the surfer deep into the pocket so the wave arches right over his shoulder/head
    start_r = n_rows - sprite_h + 1
    curve_col = 0
    for r_off in (start_r + 2):(start_r + sprite_h - 2)
        r_chars = collect(lines[canvas_rows[r_off]])
        r_braille = findall(c -> ('⠀' <= c <= '⣿'), r_chars)
        for c in length(r_braille):-1:1
            if r_chars[r_braille[c]] != '⠀'
                curve_col = max(curve_col, c)
                break
            end
        end
    end
    col_start = curve_col > 0 ?
        clamp(curve_col + 1, 2, canvas_w - sprite_w + 1) :
        max(2, canvas_w - sprite_w - 2)

    for (sr, r_offset) in enumerate(start_r:(start_r + sprite_h - 1))
        row_idx = canvas_rows[r_offset]
        chars = collect(lines[row_idx])
        braille_pos = findall(c -> ('⠀' <= c <= '⣿'), chars)
        styled_chars = String[string(c) for c in chars]

        # Find the leftmost likelihood curve column on this row so water goes right up to the curve
        first_curve_c = findfirst(c -> chars[braille_pos[c]] != '⠀', 1:length(braille_pos))
        water_start = isnothing(first_curve_c) ? 1 : first_curve_c + 1

        if sr >= sprite_h - 1
            # Extend the ocean (shades of blue only, no white at the bottom) from the likelihood curve to the right edge
            is_upper_water = (sr == sprite_h - 1)
            for col in water_start:length(braille_pos)
                c_idx = braille_pos[col]
                if chars[c_idx] == '⠀'
                    phase_c = mod(col + anim_tick, 6)
                    if is_upper_water
                        # Curl smoothly up where the water meets the likelihood curve on the left
                        bchar = (col == water_start) ? '⣿' :
                                ((col >= col_start + 1 && col <= col_start + sprite_w - 2) ? '⣿' :
                                (phase_c in (1, 2, 3) ? '⣿' : (phase_c == 4 ? '⣾' : (phase_c == 0 ? '⣷' : '⣶'))))
                        col_code = phase_c in (1, 2, 3) ? _SURFER_PALETTE['C'] : _SURFER_PALETTE['T']
                    else
                        bchar = (col == water_start) ? '⣿' : (phase_c in (1, 2, 3, 4) ? '⣿' : '⠿')
                        col_code = iseven(col + anim_tick) ? _SURFER_PALETTE['T'] : _SURFER_PALETTE['D']
                    end
                    styled_chars[c_idx] = "\e[38;2;" * col_code * "m" * string(bchar) * "\e[0m"
                end
            end
            lines[row_idx] = join(styled_chars)
        else
            # Curl the ocean water up along the lower slope of the likelihood curve (sr == 2, 3, 4)
            if !isnothing(first_curve_c) && sr >= 2
                # Find the end of the contiguous curve segment at the left slope
                slope_c = first_curve_c
                while slope_c < length(braille_pos) && chars[braille_pos[slope_c + 1]] != '⠀'
                    slope_c += 1
                end
                curl_chars = sr == 4 ? [('⣿', 'T'), ('⣷', 'C'), ('⣄', 'C')] :
                             (sr == 3 ? [('⣷', 'C'), ('⣄', 'C')] : [('⣄', 'C')])
                for (offset, (c_ch, c_pal)) in enumerate(curl_chars)
                    cc = slope_c + offset
                    if cc <= length(braille_pos) && chars[braille_pos[cc]] == '⠀'
                        styled_chars[braille_pos[cc]] = "\e[38;2;" * _SURFER_PALETTE[c_pal] * "m" * string(c_ch) * "\e[0m"
                    end
                end
            end

            # Overlay the surfer & surfboard
            if length(braille_pos) >= col_start + sprite_w - 1
                for k in 1:sprite_w
                    c_idx = braille_pos[col_start + k - 1]
                    if chars[c_idx] == '⠀'
                        bchar, col_code = cells[sr, k]
                        if bchar != '⠀'
                            styled_chars[c_idx] = "\e[38;2;" * col_code * "m" * string(bchar) * "\e[0m"
                        end
                    end
                end
            end
            lines[row_idx] = join(styled_chars)
        end
    end
    return join(lines, '\n')
end

function _render_frame!(tui::Union{Nothing, EM_TUI}, dashboard, total_w::Int)
    if !isnothing(tui) && (!isempty(tui.init_warnings) || !isempty(tui.final_warnings))
        warn_lines = String[]
        for (msg, cnt) in tui.init_warnings
            suffix = cnt > 1 ? " {dim}(×$(cnt)){/dim}" : ""
            push!(warn_lines, "  {#FFB347}⚠ [Init] {/#FFB347} $(msg)$(suffix)")
        end
        for (msg, cnt) in tui.final_warnings
            suffix = cnt > 1 ? " {dim}(×$(cnt)){/dim}" : ""
            push!(warn_lines, "  {#FF4D6D}⚠ [Final]{/#FF4D6D} $(msg)$(suffix)")
        end
        w_style = isempty(tui.final_warnings) ? "#FFA647" : "#FF4D6D"
        warn_panel = Term.Panel(join(warn_lines, "\n"); title="Warnings", style=w_style, width=total_w)
        dashboard = dashboard / warn_panel
    end
    frame_str = string(dashboard)
    if !isnothing(tui) && tui.last_lines > 0
        print(stdout, "\e[$(tui.last_lines)A\r\e[J", frame_str, "\n")
    else
        print(stdout, frame_str, "\n")
    end
    if !isnothing(tui)
        tui.last_lines = count(==('\n'), frame_str) + 1
    end
    flush(stdout)
end

function render_build!(tui::EM_TUI, stage::Int, msg::String, n_pkgs::Int=0)
    tui.build_stage = stage
    tui.build_msg = msg
    tui.render_count += 1
    if n_pkgs > 0
        tui.n_packages = n_pkgs
    end
    tui.build_elapsed = (time_ns() - tui.build_start_ns) / 1e9
    terminal = displaysize(stdout)
    total_w = clamp(terminal[2] - 2, 76, 120)
    right_w = 34
    left_w = total_w - right_w
    panel_h = clamp(Int(round(terminal[1] * 0.55)), 14, 24)

    build_frac = (stage - 0.5) / 4.0
    b_bar = _tui_bar(build_frac, 20; palette=_SUNSET_BAR)
    e_bar = _tui_bar(0.0, 20; palette=_OCEAN_BAR)
    top_text = "  {bold #FFD166}Build {/bold #FFD166} $(b_bar)  {dim}[$(stage)/4: $(msg)]{/dim}\n" *
               "  {bold #48CAE4}EM Fit{/bold #48CAE4} $(e_bar)  {dim}[Waiting for compilation]{/dim}"
    top_panel = Term.Panel(top_text; title="🌅 ParsaModel", style="#FF7E5F", width=total_w)

    steps = [
        (1, "Prime Observations (X)"),
        (2, "E-Step Conditioning Graph"),
        (3, "M-Step & π Parameter Mappings"),
        (4, "Log-Likelihood Evaluation Tree")
    ]
    checklist = IOBuffer()
    println(checklist, "")
    for (idx, label) in steps
        if idx < stage
            println(checklist, "  {#FFD166}✓{/#FFD166} $(label)")
        elseif idx == stage
            println(checklist, "  {#FF7E5F}⟳{/#FF7E5F} {bold #FFD166}$(label){/bold #FFD166}")
        else
            println(checklist, "  {dim}• $(label){/dim}")
        end
    end
    println(checklist, "\n  {dim}Elapsed build time: $(round(tui.build_elapsed, digits=2))s{/dim}")
    left_panel = Term.Panel(String(take!(checklist)); title="Structure Compilation", style="#FFA647", width=left_w, height=panel_h)

    info_text = join([
        "• {bold #90E0EF}Status:{/bold #90E0EF}     {#FFB347}Compiling{/#FFB347}",
        "• {bold #90E0EF}Stage:{/bold #90E0EF}      $(stage) / 4",
        "• {bold #90E0EF}Obs (N):{/bold #90E0EF}    $(tui.n_obs)",
        "• {bold #90E0EF}Packages:{/bold #90E0EF}   $(tui.n_packages > 0 ? string(tui.n_packages) : "...")",
        "• {bold #90E0EF}Restarts:{/bold #90E0EF}   0 / $(tui.n_init)",
        "• {bold #90E0EF}Init ε:{/bold #90E0EF}     $(round(tui.init_eps, sigdigits=2))",
        "• {bold #90E0EF}Final ε:{/bold #90E0EF}    $(round(tui.eps, sigdigits=2))",
        "• {bold #90E0EF}Elapsed:{/bold #90E0EF}    $(round(tui.build_elapsed, digits=2))s"
    ], "\n")
    right_panel = Term.Panel(info_text; title="Model & Convergence", style="#0077B6", width=right_w, height=panel_h)

    _render_frame!(tui, top_panel / (left_panel * right_panel), total_w)
end

function finish_build!(tui::EM_TUI, n_pkgs::Int)
    tui.n_packages = n_pkgs
    tui.build_stage = 4
    tui.build_elapsed = (time_ns() - tui.build_start_ns) / 1e9
    tui.em_start_ns = time_ns()
end

function plotit(lines, final_lines;
                tui::Union{Nothing, EM_TUI}=nothing,
                phase::Symbol=:init,
                i_init::Int=1,
                step::Int=1,
                max_steps::Int=10000,
                rel_diff::Real=NaN,
                target_eps::Real=1e-6,
                force::Bool=false)
    now_ns = time_ns()
    if !isnothing(tui)
        tui.total_iters += 1
        if !force && step > 1 && (now_ns - tui.last_render_ns) < 33_000_000
            return nothing
        end
        tui.last_render_ns = now_ns
        tui.render_count += 1
    end

    terminal = displaysize(stdout)
    total_w = clamp(terminal[2] - 2, 76, 120)
    right_w = 34
    left_w = total_w - right_w
    panel_h = clamp(Int(round(terminal[1] * 0.62)), 16, 26)
    plot_w = max(16, left_w - 18)
    plot_h = max(6, panel_h - 6)

    lll = [Float64(l) for l in [reduce(vcat, lines); final_lines] if l != 0 && isfinite(l)]
    if isempty(lll)
        lll = [0.0]
    end
    ymax = maximum(lll)
    ymin = minimum(lll)
    if ymin == ymax
        ymin -= 1.0
        ymax += 1.0
    end
    xmin = 0

    best_init_idx = 1
    best_init_ll = -Inf
    for (idx, l) in enumerate(lines)
        if !isempty(l)
            m_val = maximum(l)
            if m_val > best_init_ll
                best_init_ll = m_val
                best_init_idx = idx
            end
        end
    end
    xmax_init = isempty(lines[best_init_idx]) ? 1 : length(lines[best_init_idx])
    max_any_init = maximum([length(l) for l in lines]; init=1)
    xmax = max(max_any_init, xmax_init + length(final_lines))

    plt = UnicodePlots.lineplot([0], [ymin];
        ylim=(round(ymin, digits=1), round(ymax, digits=1)),
        xlim=(xmin, xmax),
        xlabel="steps",
        margin=1,
        padding=0,
        height=plot_h,
        width=plot_w,
        color=:blue)

    for i in 1:size(lines)[1]
        n_l = lines[i]
        if !isempty(n_l) && i != best_init_idx && i != i_init
            UnicodePlots.lineplot!(plt, 1:length(n_l), Float64.(n_l), color=:blue)
        end
    end
    if !isempty(lines[best_init_idx])
        UnicodePlots.lineplot!(plt, 1:length(lines[best_init_idx]), Float64.(lines[best_init_idx]), color=:red)
    end
    if phase == :init && i_init <= length(lines) && !isempty(lines[i_init]) && i_init != best_init_idx
        UnicodePlots.lineplot!(plt, 1:length(lines[i_init]), Float64.(lines[i_init]), color=:magenta)
    end
    if length(final_lines) > 0
        n_l = Float64.(final_lines)
        UnicodePlots.lineplot!(plt, xmax_init:(length(n_l)+xmax_init-1), n_l, color=:yellow)
    end

    str_plt = sprint(show, MIME("text/plain"), plt; context=:color=>true)
    str_plt = replace(str_plt, _PLOT_RGB_MAP...)
    wave_frame = !isnothing(tui) ? tui.render_count : step
    str_plt = _overlay_surfer_in_plot(str_plt, wave_frame)
    left_panel = Term.Panel(str_plt; title="🌊 Log-Likelihood", style="#00B4DB", width=left_w, height=panel_h)

    curr_ll = !isempty(final_lines) ? Float64(final_lines[end]) :
              (i_init <= length(lines) && !isempty(lines[i_init]) ? Float64(lines[i_init][end]) : best_init_ll)

    n_init_total = !isnothing(tui) ? tui.n_init : length(lines)
    if phase == :init
        em_frac = (i_init - 0.5) / max(1, n_init_total) * 0.75
        em_status = "Init $(i_init)/$(n_init_total) • Step $(step)"
        phase_str = "{#FF9966}Init ($(i_init)/$(n_init_total)){/#FF9966}"
    elseif phase == :final
        init_e = !isnothing(tui) ? tui.init_eps : 1e-3
        final_e = !isnothing(tui) ? tui.eps : 1e-6
        conv_frac = (isfinite(rel_diff) && rel_diff > 0) ?
            clamp((log10(init_e) - log10(rel_diff)) / max(1e-9, log10(init_e) - log10(final_e)), 0.0, 0.99) : 0.5
        em_frac = 0.75 + 0.25 * conv_frac
        em_status = "Refining • Step $(step)"
        phase_str = "{#48CAE4}Final EM{/#48CAE4}"
    else
        em_frac = 1.0
        em_status = "Converged at step $(step)"
        phase_str = "{#FFD166}Converged ☀{/#FFD166}"
    end

    build_time_str = !isnothing(tui) ? "$(round(tui.build_elapsed, digits=2))s" : "done"
    pkgs_str = (!isnothing(tui) && tui.n_packages > 0) ? "$(tui.n_packages) pkgs" : "ready"
    b_bar = _tui_bar(1.0, 20; palette=_SUNSET_BAR)
    e_bar = _tui_bar(em_frac, 20; palette=_OCEAN_BAR)
    top_text = "  {bold #FFD166}Build {/bold #FFD166} $(b_bar)  {dim}[Done in $(build_time_str) • $(pkgs_str)]{/dim}\n" *
               "  {bold #48CAE4}EM Fit{/bold #48CAE4} $(e_bar)  {dim}[$(em_status)]{/dim}"
    top_panel = Term.Panel(top_text; title="🌅 ParsaModel", style="#FF7E5F", width=total_w)

    em_elapsed = !isnothing(tui) ? (now_ns - tui.em_start_ns) / 1e9 : 0.0
    speed = ( !isnothing(tui) && em_elapsed > 0 ) ? Int(round(tui.total_iters / em_elapsed)) : 0
    rel_str = isfinite(rel_diff) ? string(round(rel_diff, sigdigits=3)) : "..."

    info_lines = [
        "• {bold #90E0EF}Phase:{/bold #90E0EF}      $(phase_str)",
        "• {bold #90E0EF}Step:{/bold #90E0EF}       $(step) / $(max_steps)",
        "• {bold #90E0EF}Best Init:{/bold #90E0EF}  #$(best_init_idx) / $(n_init_total)",
        "• {bold #90E0EF}Best LL:{/bold #90E0EF}    {#FFD166}$(round(max(best_init_ll, curr_ll), digits=4)){/#FFD166}",
        "• {bold #90E0EF}Curr LL:{/bold #90E0EF}    $(round(curr_ll, digits=4))",
        "• {bold #90E0EF}ΔLL / |LL|:{/bold #90E0EF} $(rel_str)",
        "• {bold #90E0EF}Target ε:{/bold #90E0EF}   $(round(target_eps, sigdigits=2))",
        "• {bold #90E0EF}Obs (N):{/bold #90E0EF}    $(!isnothing(tui) ? string(tui.n_obs) : "...")",
        "• {bold #90E0EF}Packages:{/bold #90E0EF}   $(!isnothing(tui) ? string(tui.n_packages) : "...")",
        "• {bold #90E0EF}Speed:{/bold #90E0EF}      $(speed) iter/s",
        "• {bold #90E0EF}Elapsed:{/bold #90E0EF}    $(round(em_elapsed, digits=2))s"
    ]
    right_panel = Term.Panel(join(info_lines, "\n"); title="Model & Convergence", style="#0077B6", width=right_w, height=panel_h)

    _render_frame!(tui, top_panel / (left_panel * right_panel), total_w)
end
