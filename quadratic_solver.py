import cmath
import sys


def solve_quadratic(a, b, c):
    """Return the two solutions to ax^2 + bx + c = 0."""
    if a == 0:
        raise ValueError("Coefficient 'a' must not be zero for a quadratic equation")
    discriminant = cmath.sqrt(b * b - 4 * a * c)
    root1 = (-b + discriminant) / (2 * a)
    root2 = (-b - discriminant) / (2 * a)
    return root1, root2


def main(args):
    if len(args) != 3:
        print("Usage: python quadratic_solver.py a b c")
        return
    try:
        a = float(args[0])
        b = float(args[1])
        c = float(args[2])
    except ValueError:
        print("Coefficients must be numbers")
        return
    try:
        roots = solve_quadratic(a, b, c)
        print(f"Roots: {roots[0]} , {roots[1]}")
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main(sys.argv[1:])
