def clean_coefficients(coeffs, tol=1e-9):
    """
    Zero out coefficients whose magnitude is below *tol* × max(|coeffs|).

    Prevents floating-point noise (e.g. 2e-15) from appearing as spurious
    terms when displaying computed transfer-function polynomials.

    Args:
        coeffs: list or array of polynomial coefficients
        tol: relative tolerance (default 1e-9)

    Returns:
        list of cleaned coefficients
    """
    if not list(coeffs):
        return list(coeffs)
    arr = [float(c) for c in coeffs]
    max_val = max(abs(c) for c in arr)
    if max_val == 0:
        return arr
    return [0.0 if abs(c) < tol * max_val else c for c in arr]


def _fmt_coeff(value):
    """
    Format a single coefficient as a LaTeX-safe string.

    Uses integer representation when exact, otherwise up to 6 significant
    figures with trailing zeros stripped — never produces scientific notation.
    """
    f = float(value)
    # Exact integer (within float precision)
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    # Up to 6 significant figures, strip trailing zeros
    formatted = f"{f:.6g}"
    # :g can still emit scientific notation for very large/small values;
    # fall back to fixed notation in that case.
    if "e" in formatted or "E" in formatted:
        formatted = f"{f:.6f}".rstrip("0").rstrip(".")
    return formatted


def format_polynomial(coeffs, variable="s"):
    """
    Render a polynomial as a LaTeX string.

    Handles negative coefficients properly (e.g., "s^{2} - 2s + 1" instead of
    "s^{2} + -2s + 1"), suppresses the coefficient 1 on non-constant terms,
    and never emits scientific notation.

    Args:
        coeffs: list of coefficients in descending power order
        variable: variable name (default "s")

    Returns:
        str: LaTeX polynomial string
    """
    if not list(coeffs):
        return "0"

    terms: list[tuple[str, str]] = []  # (sign, abs_term_string)
    degree = len(coeffs) - 1

    for i, coeff in enumerate(coeffs):
        if coeff == 0:
            continue

        power = degree - i
        abs_c = abs(coeff)
        sign = "-" if coeff < 0 else "+"

        if power == 0:
            term_str = _fmt_coeff(abs_c)
        elif power == 1:
            term_str = f"{_fmt_coeff(abs_c)}{variable}" if abs_c != 1 else variable
        else:
            term_str = (
                f"{_fmt_coeff(abs_c)}{variable}^{{{power}}}"
                if abs_c != 1
                else f"{variable}^{{{power}}}"
            )

        terms.append((sign, term_str))

    if not terms:
        return "0"

    # First term: omit the leading "+"
    first_sign, first_term = terms[0]
    result = f"-{first_term}" if first_sign == "-" else first_term

    for sign, term_str in terms[1:]:
        result += f" {sign} {term_str}"

    return result


def format_metric(value, fmt=".3f", fallback="N/A"):
    """
    Safely format a numeric value; return *fallback* if conversion fails.

    Args:
        value: value to format
        fmt: format spec string (default ".3f")
        fallback: string to return on failure (default "N/A")

    Returns:
        str
    """
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return fallback
