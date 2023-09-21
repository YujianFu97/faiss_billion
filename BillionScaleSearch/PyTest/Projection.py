import math

def find_height_from_edge_lengths(a, b, c, d, e, f):
    # Volume of the tetrahedron using Cayley-Menger determinant
    V = (1/12) * math.sqrt((a**2 * b**2 * c**2) + 2*(a**2 * b**2 * d**2 + a**2 * c**2 * e**2 + b**2 * c**2 * f**2)
                            - (a**4 * b**2 + b**4 * c**2 + c**4 * a**2 + a**2 * d**4 + b**2 * e**4 + c**2 * f**4))
    
    # Semiperimeter of base triangle ABC
    s = (a + b + c) / 2
    
    # Area of the base triangle using Heron's formula
    S = math.sqrt(s * (s - a) * (s - b) * (s - c))
    
    # Height of the vertex D with respect to the base ABC
    h = (3 * V) / S
    
    return h

# Example edge lengths (in arbitrary units)
a = 3
b = 4
c = 5
d = 6
e = 7
f = 8

height = find_height_from_edge_lengths(a, b, c, d, e, f)
print("Height of vertex D with respect to base ABC:", height)
