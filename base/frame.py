import jax.numpy as jnp
from dataclasses import dataclass

@dataclass(frozen=True)
class Frame:
    x: jnp.ndarray  # shape (3,)
    y: jnp.ndarray  # shape (3,)
    z: jnp.ndarray  # shape (3,)

    def to_local(self, v: jnp.ndarray) -> jnp.ndarray:
        """Convert a vector v from world space to the local frame."""
        return jnp.array([
            jnp.dot(v, self.x),
            jnp.dot(v, self.y),
            jnp.dot(v, self.z)
        ])

    def from_local(self, v: jnp.ndarray) -> jnp.ndarray:
        """Convert a vector v from local space to world space."""
        return self.x * v[0] + self.y * v[1] + self.z * v[2]

def create_frame(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> Frame:
    """Create a frame after verifying orthogonality and normalization."""
    tol = 1e-4
    if not (jnp.abs(jnp.linalg.norm(x) - 1.0) < tol):
        raise ValueError(f"x is not normalized: {jnp.linalg.norm(x)}")
    if not (jnp.abs(jnp.linalg.norm(y) - 1.0) < tol):
        raise ValueError(f"y is not normalized: {jnp.linalg.norm(y)}")
    if not (jnp.abs(jnp.linalg.norm(z) - 1.0) < tol):
        raise ValueError(f"z is not normalized: {jnp.linalg.norm(z)}")
    if not (jnp.abs(jnp.dot(x, y)) < tol):
        raise ValueError("x and y are not orthogonal")
    if not (jnp.abs(jnp.dot(y, z)) < tol):
        raise ValueError("y and z are not orthogonal")
    if not (jnp.abs(jnp.dot(z, x)) < tol):
        raise ValueError("z and x are not orthogonal")
    return Frame(x=x, y=y, z=z)

def copysign(x: float, y: float) -> float:
    """Return |x| if y is nonnegative, else -|x|."""
    return jnp.abs(x) if y >= 0 else -jnp.abs(x)

def coordinate_system(v: jnp.ndarray) -> (jnp.ndarray, jnp.ndarray):
    """Generate two vectors that form an orthonormal basis with v."""
    sign = copysign(1.0, v[2])
    a = -1.0 / (sign + v[2])
    b = v[0] * v[1] * a
    v2 = jnp.array([1.0 + sign * v[0] * v[0] * a, sign * b, -sign * v[0]])
    v3 = jnp.array([b, sign + v[1] * v[1] * a, -v[1]])
    return v2 / jnp.linalg.norm(v2), v3 / jnp.linalg.norm(v3)

def frame_from_xz(x: jnp.ndarray, z: jnp.ndarray) -> Frame:
    """Create a frame from x and z vectors (re-orthogonalize x with respect to z)."""
    x = x / jnp.linalg.norm(x)
    z = z / jnp.linalg.norm(z)
    x = x - jnp.dot(x, z) * z
    x = x / jnp.linalg.norm(x)
    y = jnp.cross(z, x)
    return create_frame(x, y, z)

def frame_from_xy(x: jnp.ndarray, y: jnp.ndarray) -> Frame:
    x = x / jnp.linalg.norm(x)
    y = y / jnp.linalg.norm(y)
    z = jnp.cross(x, y)
    return create_frame(x, y, z)

def frame_from_z(z: jnp.ndarray) -> Frame:
    z = z / jnp.linalg.norm(z)
    x, y = coordinate_system(z)
    return create_frame(x, y, z)

def frame_from_x(x: jnp.ndarray) -> Frame:
    x = x / jnp.linalg.norm(x)
    y, z = coordinate_system(x)
    return create_frame(x, y, z)

def frame_from_y(y: jnp.ndarray) -> Frame:
    y = y / jnp.linalg.norm(y)
    z, x = coordinate_system(y)
    return create_frame(x, y, z)
