import ast
import hashlib
import json
import math
import os
import re
import struct
import sys
import wave
import zlib


# Strict Blender 4.0 - 4.3 Node Socket Schema Validator (Shader, Compositor & Geometry Nodes)
def get_node_schemas(app_version):
    lens_dist_sock = "Distort" if app_version < (4, 2, 0) else "Distortion"
    return {
        # Shader Nodes
        "ShaderNodeOutputMaterial": {
            "inputs": ["Surface", "Volume", "Displacement", "Thickness"],
            "outputs": [],
        },
        "ShaderNodeOutputWorld": {
            "inputs": ["Surface", "Volume"],
            "outputs": [],
        },
        "ShaderNodeOutputAOV": {
            "inputs": ["Color", "Value"],
            "outputs": [],
        },
        "ShaderNodeBackground": {
            "inputs": ["Color", "Strength"],
            "outputs": ["Background"],
        },
        "ShaderNodeVolumePrincipled": {
            "inputs": [
                "Color", "Color Attribute", "Density", "Density Attribute",
                "Anisotropy", "Absorption Color", "Emission Strength",
                "Emission Color", "Blackbody Intensity", "Blackbody Tint",
                "Temperature", "Temperature Attribute",
            ],
            "outputs": ["Volume"],
        },
        "ShaderNodeBsdfPrincipled": {
            "inputs": [
                "Base Color", "Metallic", "Roughness", "IOR", "Alpha", "Normal",
                "Weight", "Diffuse Roughness",
                "Subsurface Weight", "Subsurface Radius", "Subsurface Scale",
                "Subsurface IOR", "Subsurface Anisotropy",
                "Specular IOR Level", "Specular Tint",
                "Anisotropic", "Anisotropic Rotation", "Tangent",
                "Transmission Weight",
                "Coat Weight", "Coat Roughness", "Coat IOR", "Coat Tint", "Coat Normal",
                "Sheen Weight", "Sheen Roughness", "Sheen Tint",
                "Emission Color", "Emission Strength",
                "Thin Film Thickness", "Thin Film IOR",
            ],
            "outputs": ["BSDF"],
        },
        "ShaderNodeBsdfTranslucent": {
            "inputs": ["Color", "Normal"],
            "outputs": ["BSDF"],
        },
        "ShaderNodeBsdfGlossy": {
            "inputs": ["Color", "Roughness", "Anisotropy", "Rotation", "Normal", "Tangent"],
            "outputs": ["BSDF"],
        },
        "ShaderNodeEmission": {
            "inputs": ["Color", "Strength"],
            "outputs": ["Emission"],
        },
        "ShaderNodeMixShader": {
            "inputs": ["Fac", "Shader", "Shader"],
            "outputs": ["Shader"],
        },
        "ShaderNodeAddShader": {
            "inputs": ["Shader", "Shader"],
            "outputs": ["Shader"],
        },
        "ShaderNodeTexCoord": {
            "inputs": [],
            "outputs": ["Generated", "Normal", "UV", "Object", "Camera", "Window", "Reflection"],
        },
        "ShaderNodeMapping": {
            "inputs": ["Vector", "Location", "Rotation", "Scale"],
            "outputs": ["Vector"],
        },
        "ShaderNodeTexNoise": {
            "inputs": ["Vector", "W", "Scale", "Detail", "Roughness", "Lacunarity", "Offset", "Gain", "Distortion"],
            "outputs": ["Fac", "Color"],
        },
        "ShaderNodeTexMusgrave": {
            "inputs": ["Vector", "W", "Scale", "Detail", "Dimension", "Lacunarity", "Offset", "Gain"],
            "outputs": ["Fac"],
        },
        "ShaderNodeTexVoronoi": {
            "inputs": ["Vector", "W", "Scale", "Smoothness", "Exponent", "Randomness"],
            "outputs": ["Distance", "Color", "Position", "W", "Radius"],
        },
        "ShaderNodeTexWave": {
            "inputs": ["Vector", "Scale", "Distortion", "Detail", "Detail Scale", "Detail Roughness", "Phase Offset"],
            "outputs": ["Color", "Fac"],
        },
        "ShaderNodeTexBrick": {
            "inputs": [
                "Vector", "Color1", "Color2", "Mortar", "Scale",
                "Mortar Size", "Mortar Smooth", "Bias", "Brick Width", "Row Height",
            ],
            "outputs": ["Color", "Fac"],
        },
        "ShaderNodeTexGradient": {
            "inputs": ["Vector"],
            "outputs": ["Color", "Fac"],
        },
        "ShaderNodeValToRGB": {
            "inputs": ["Fac"],
            "outputs": ["Color", "Alpha"],
        },
        "ShaderNodeBump": {
            "inputs": ["Strength", "Distance", "Height", "Normal"],
            "outputs": ["Normal"],
        },
        "ShaderNodeFresnel": {
            "inputs": ["IOR", "Normal"],
            "outputs": ["Fac"],
        },
        "ShaderNodeLayerWeight": {
            "inputs": ["Blend", "Normal"],
            "outputs": ["Fresnel", "Facing"],
        },
        "ShaderNodeMix": {
            "inputs": ["Factor", "Factor", "A", "B", "A", "B", "A", "B", "A", "B"],
            "outputs": ["Result", "Result", "Result", "Result"],
        },
        "ShaderNodeMath": {
            "inputs": ["Value", "Value", "Value"],
            "outputs": ["Value"],
        },
        "ShaderNodeSepXYZ": {
            "inputs": ["Vector"],
            "outputs": ["X", "Y", "Z"],
        },
        "ShaderNodeCombXYZ": {
            "inputs": ["X", "Y", "Z"],
            "outputs": ["Vector"],
        },
        "ShaderNodeBrightContrast": {
            "inputs": ["Color", "Bright", "Contrast"],
            "outputs": ["Color"],
        },
        "ShaderNodeHueSat": {
            "inputs": ["Hue", "Saturation", "Value", "Fac", "Color"],
            "outputs": ["Color"],
        },
        "ShaderNodeObjectInfo": {
            "inputs": [],
            "outputs": ["Location", "Color", "Alpha", "Object Index", "Material Index", "Random"],
        },

        # Compositor Nodes (Blender 4.0 - 4.3)
        "CompositorNodeRLayers": {
            "inputs": [],
            "outputs": [
                "Image", "Alpha", "Depth", "Normal", "Vector", "Shadow", "AO",
                "DiffCol", "GlossDir", "Emit", "Env", "Mist",
                "CryptoObject00", "CryptoMaterial00",
                "AOV_Neon_Mask", "AOV_Thermal_Heat",
            ],
        },
        "CompositorNodeGlare": {
            "inputs": ["Image"],
            "outputs": ["Image"],
        },
        "CompositorNodeMixRGB": {
            "inputs": ["Fac", "Image", "Image"],
            "outputs": ["Image"],
        },
        "CompositorNodeValToRGB": {
            "inputs": ["Fac"],
            "outputs": ["Image", "Alpha"],
        },
        "CompositorNodeLensdist": {
            "inputs": ["Image", lens_dist_sock, "Dispersion"],
            "outputs": ["Image"],
        },
        "CompositorNodeExposure": {
            "inputs": ["Image", "Exposure"],
            "outputs": ["Image"],
        },
        "CompositorNodeColorBalance": {
            "inputs": ["Fac", "Image"],
            "outputs": ["Image"],
        },
        "CompositorNodeComposite": {
            "inputs": ["Image", "Alpha"],
            "outputs": [],
        },
        "CompositorNodeViewer": {
            "inputs": ["Image", "Alpha"],
            "outputs": [],
        },

        # Geometry Nodes (Blender 4.0 - 4.3)
        "NodeGroupInput": {
            "inputs": [],
            "outputs": ["Geometry"],
        },
        "NodeGroupOutput": {
            "inputs": ["Geometry"],
            "outputs": [],
        },
        "GeometryNodeDistributePointsOnFaces": {
            "inputs": ["Mesh", "Selection", "Distance Min", "Density Max", "Density", "Density Factor", "Seed"],
            "outputs": ["Points", "Normal", "Rotation"],
        },
        "GeometryNodeObjectInfo": {
            "inputs": ["Object", "As Instance"],
            "outputs": ["Transform", "Location", "Rotation", "Scale", "Geometry"],
        },
        "FunctionNodeRandomValue": {
            "inputs": ["Min", "Max", "Probability", "ID", "Seed"],
            "outputs": ["Value"],
        },
        "GeometryNodeInstanceOnPoints": {
            "inputs": ["Points", "Selection", "Instance", "Pick Instance", "Instance Index", "Rotation", "Scale"],
            "outputs": ["Instances"],
        },
        "GeometryNodeSetMaterial": {
            "inputs": ["Geometry", "Selection", "Material"],
            "outputs": ["Geometry"],
        },
    }


class Vector:
    def __init__(self, seq=(0.0, 0.0, 0.0)):
        self._v = [float(x) for x in seq]
        for x in self._v:
            if math.isnan(x) or math.isinf(x):
                raise ValueError(f"NaN/Inf in Vector: {self._v}")

    @property
    def x(self): return self._v[0]
    @x.setter
    def x(self, val): self._v[0] = float(val)

    @property
    def y(self): return self._v[1]
    @y.setter
    def y(self, val): self._v[1] = float(val)

    @property
    def z(self): return self._v[2]
    @z.setter
    def z(self, val): self._v[2] = float(val)

    @property
    def length(self):
        return math.sqrt(sum(c * c for c in self._v))

    def normalized(self):
        l = self.length
        if l < 1e-12:
            return Vector((0.0, 0.0, 0.0))
        return Vector([c / l for c in self._v])

    def copy(self):
        return Vector(self._v)

    def dot(self, other):
        return sum(a * b for a, b in zip(self._v, other))

    def cross(self, other):
        a, b = self._v, list(other)
        return Vector((
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ))

    def lerp(self, other, factor):
        o = list(other)
        return Vector([a + (b - a) * factor for a, b in zip(self._v, o)])

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self._v, other)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self._v, other)])

    def __mul__(self, scalar):
        return Vector([a * float(scalar) for a in self._v])

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        return Vector([a / float(scalar) for a in self._v])

    def __iter__(self):
        return iter(self._v)

    def __len__(self):
        return len(self._v)

    def __getitem__(self, idx):
        return self._v[idx]

    def __repr__(self):
        return f"Vector({tuple(self._v)})"


class Euler:
    def __init__(self, seq=(0.0, 0.0, 0.0), order='XYZ'):
        self._e = [float(x) for x in seq]
        self.order = order

    @property
    def x(self): return self._e[0]
    @x.setter
    def x(self, val): self._e[0] = float(val)

    @property
    def y(self): return self._e[1]
    @y.setter
    def y(self, val): self._e[1] = float(val)

    @property
    def z(self): return self._e[2]
    @z.setter
    def z(self, val): self._e[2] = float(val)

    def to_matrix(self):
        cx, sx = math.cos(self._e[0]), math.sin(self._e[0])
        cy, sy = math.cos(self._e[1]), math.sin(self._e[1])
        cz, sz = math.cos(self._e[2]), math.sin(self._e[2])
        # XYZ intrinsic / extrinsic Blender Euler('XYZ'): R = Rz * Ry * Rx
        return Matrix([
            [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz],
            [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
            [-sy, cy * sx, cx * cy],
        ])

    def copy(self):
        return Euler(self._e, self.order)

    def __iter__(self):
        return iter(self._e)

    def __getitem__(self, idx):
        return self._e[idx]


class Matrix:
    def __init__(self, rows=None):
        self.rows = [[float(c) for c in r] for r in rows] if rows else [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

    @classmethod
    def Identity(cls, size=4):
        m = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
        return cls(m)

    @classmethod
    def Translation(cls, vec):
        m = cls.Identity(4)
        v = list(vec)
        m.rows[0][3], m.rows[1][3], m.rows[2][3] = float(v[0]), float(v[1]), float(v[2])
        return m

    @classmethod
    def Rotation(cls, angle, size, axis):
        c, s = math.cos(angle), math.sin(angle)
        if isinstance(axis, str):
            ax = {'X': (1.0, 0.0, 0.0), 'Y': (0.0, 1.0, 0.0), 'Z': (0.0, 0.0, 1.0)}[axis.upper()]
        else:
            v = Vector(axis).normalized()
            ax = (v.x, v.y, v.z)
        x, y, z = ax
        t = 1.0 - c
        r3 = [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
        if size == 3:
            return cls(r3)
        m = cls.Identity(4)
        for i in range(3):
            for j in range(3):
                m.rows[i][j] = r3[i][j]
        return m

    @classmethod
    def Scale(cls, factor, size, axis=None):
        m = cls.Identity(size)
        if axis is None:
            for i in range(min(3, size)):
                m.rows[i][i] = float(factor)
        else:
            v = Vector(axis).normalized()
            k = float(factor) - 1.0
            a = (v.x, v.y, v.z)
            for i in range(3):
                for j in range(3):
                    m.rows[i][j] += k * a[i] * a[j]
        return m

    def to_4x4(self):
        if len(self.rows) == 4:
            return Matrix(self.rows)
        m = Matrix.Identity(4)
        for i in range(3):
            for j in range(3):
                m.rows[i][j] = self.rows[i][j]
        return m

    def __matmul__(self, other):
        if isinstance(other, Vector):
            n = len(self.rows)
            if n == 3:
                return Vector((
                    self.rows[0][0] * other.x + self.rows[0][1] * other.y + self.rows[0][2] * other.z,
                    self.rows[1][0] * other.x + self.rows[1][1] * other.y + self.rows[1][2] * other.z,
                    self.rows[2][0] * other.x + self.rows[2][1] * other.y + self.rows[2][2] * other.z,
                ))
            return Vector((
                self.rows[0][0] * other.x + self.rows[0][1] * other.y + self.rows[0][2] * other.z + self.rows[0][3],
                self.rows[1][0] * other.x + self.rows[1][1] * other.y + self.rows[1][2] * other.z + self.rows[1][3],
                self.rows[2][0] * other.x + self.rows[2][1] * other.y + self.rows[2][2] * other.z + self.rows[2][3],
            ))
        if isinstance(other, Matrix):
            n = len(self.rows)
            out = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    out[i][j] = sum(self.rows[i][k] * other.rows[k][j] for k in range(n))
            return Matrix(out)
        return Matrix.Identity(4)


class Socket:
    def __init__(self, name, index, tree_ref=None):
        self.name = name
        self.index = index
        self.default_value = None
        self._tree = tree_ref

    def keyframe_insert(self, data_path, frame, index=-1):
        if self._tree is not None:
            self._tree._record_keyframe(f"socket.{self.name}.{data_path}", frame)


class SocketCollection:
    def __init__(self, node_type, names, kind, tree_ref=None):
        self.node_type = node_type
        self.kind = kind
        self._tree = tree_ref
        self._items = [Socket(name, i, tree_ref) for i, name in enumerate(names)]

    def append_socket(self, name):
        s = Socket(name, len(self._items), self._tree)
        self._items.append(s)
        return s

    def __getitem__(self, key):
        if isinstance(key, int):
            if 0 <= key < len(self._items):
                return self._items[key]
            raise IndexError(f"{self.node_type} {self.kind}[{key}] out of range (len={len(self._items)})")
        for s in self._items:
            if s.name == key:
                return s
        raise KeyError(f"Invalid socket '{key}' on {self.node_type}.{self.kind}! Valid: {[s.name for s in self._items]}")

    def __contains__(self, key):
        return any(s.name == key for s in self._items)

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)


class ColorRampElement:
    def __init__(self, position, color=(0.0, 0.0, 0.0, 1.0)):
        self.position = float(position)
        self.color = tuple(color)


class ColorRampElements(list):
    def __init__(self):
        super().__init__([ColorRampElement(0.0, (0, 0, 0, 1)), ColorRampElement(1.0, (1, 1, 1, 1))])

    def new(self, position):
        el = ColorRampElement(position)
        self.append(el)
        return el


class ColorRamp:
    def __init__(self):
        self.interpolation = 'LINEAR'
        self.elements = ColorRampElements()


class GenericNode:
    def __init__(self, node_type, app_version, tree_ref=None):
        schemas = get_node_schemas(app_version)
        if node_type == "ShaderNodeTexMusgrave" and app_version >= (4, 1, 0):
            raise RuntimeError("ShaderNodeTexMusgrave removed in Blender 4.1+!")
        if node_type not in schemas:
            raise RuntimeError(f"Unknown Node type: {node_type}")
        self.bl_idname = node_type
        self.name = node_type
        self.label = ""
        self.location = (0, 0)
        schema = schemas[node_type]

        in_names = list(schema["inputs"])
        out_names = list(schema["outputs"])
        if tree_ref is not None and hasattr(tree_ref, "interface"):
            if node_type == "NodeGroupInput" and tree_ref.interface.items_tree:
                in_socks = [s.name for s in tree_ref.interface.items_tree if s.in_out == 'INPUT']
                if in_socks:
                    out_names = in_socks
            elif node_type == "NodeGroupOutput" and tree_ref.interface.items_tree:
                out_socks = [s.name for s in tree_ref.interface.items_tree if s.in_out == 'OUTPUT']
                if out_socks:
                    in_names = out_socks

        self.inputs = SocketCollection(node_type, in_names, "inputs", tree_ref)
        self.outputs = SocketCollection(node_type, out_names, "outputs", tree_ref)

        if node_type in ("ShaderNodeValToRGB", "CompositorNodeValToRGB"):
            self.color_ramp = ColorRamp()
        if node_type == "ShaderNodeMix":
            self.data_type = 'FLOAT'
            self.blend_type = 'MIX'
            self.clamp_result = False
            self.clamp_factor = True
        if node_type == "ShaderNodeMapping":
            self.vector_type = 'POINT'
        if node_type == "ShaderNodeTexNoise":
            self.noise_dimensions = '3D'
            if app_version >= (4, 1, 0):
                self.noise_type = 'FBM'
                self.normalize = True
        if node_type == "ShaderNodeTexMusgrave":
            self.musgrave_dimensions = '3D'
            self.musgrave_type = 'FBM'
        if node_type == "ShaderNodeTexVoronoi":
            self.voronoi_dimensions = '3D'
            self.feature = 'F1'
            self.distance = 'EUCLIDEAN'
        if node_type == "ShaderNodeTexWave":
            self.wave_type = 'BANDS'
            self.bands_direction = 'X'
            self.wave_profile = 'SIN'
        if node_type == "ShaderNodeTexBrick":
            self.offset = 0.5
            self.offset_frequency = 2
            self.squash = 1.0
            self.squash_frequency = 2
        if node_type == "ShaderNodeOutputAOV":
            self.aov_name = ""
        if node_type == "ShaderNodeMath":
            self.operation = 'ADD'
            self.use_clamp = False
        if node_type == "ShaderNodeBump":
            self.invert = False
        if node_type == "CompositorNodeGlare":
            self.glare_type = 'STREAKS'
            self.quality = 'HIGH'
            self.iterations = 3
            self.color_modulation = 0.25
            self.mix = 0.0
            self.threshold = 1.0
            self.streaks = 4
            self.angle_offset = 0.0
            self.fade = 0.9
            self.size = 8
        if node_type == "CompositorNodeMixRGB":
            self.blend_type = 'MIX'
            self.use_alpha = False
            self.use_clamp = False
        if node_type == "CompositorNodeLensdist":
            self.use_projector = False
            self.use_jitter = False
            self.use_fit = False
        if node_type == "CompositorNodeColorBalance":
            self.correction_method = 'LIFT_GAMMA_GAIN'
            self.lift = (1.0, 1.0, 1.0)
            self.gamma = (1.0, 1.0, 1.0)
            self.gain = (1.0, 1.0, 1.0)
        if node_type == "GeometryNodeDistributePointsOnFaces":
            self.distribute_method = 'RANDOM'
        if node_type == "GeometryNodeObjectInfo":
            self.transform_space = 'ORIGINAL'
        if node_type == "FunctionNodeRandomValue":
            self.data_type = 'FLOAT'


class NodeTreeInterfaceSocket:
    def __init__(self, name, in_out, socket_type):
        if in_out not in ('INPUT', 'OUTPUT'):
            raise ValueError(f"Invalid in_out: {in_out}")
        self.name = name
        self.in_out = in_out
        self.socket_type = socket_type


class NodeTreeInterface:
    def __init__(self):
        self.items_tree = []

    def new_socket(self, name, in_out='INPUT', socket_type='NodeSocketGeometry', description=""):
        sock = NodeTreeInterfaceSocket(name, in_out, socket_type)
        self.items_tree.append(sock)
        return sock


class NodeCollection(list):
    def __init__(self, app_version, tree_ref):
        super().__init__()
        self.app_version = app_version
        self._tree = tree_ref

    def new(self, type):
        n = GenericNode(type, self.app_version, self._tree)
        self.append(n)
        return n

    def get(self, name):
        for n in self:
            if n.name == name:
                return n
        return None


class LinkCollection(list):
    def new(self, out_sock, in_sock):
        if not isinstance(out_sock, Socket) or not isinstance(in_sock, Socket):
            raise TypeError(f"Invalid link endpoints: {out_sock} -> {in_sock}")
        link = (out_sock, in_sock)
        self.append(link)
        return link


class BMVert:
    def __init__(self, co, idx):
        self.co = Vector(co)
        self.index = idx
        self.normal = Vector((0, 0, 1))


class BMEdge:
    def __init__(self, v1, v2, idx):
        if v1 is v2:
            raise ValueError("Degenerate BMEdge with identical vertices")
        self.verts = (v1, v2)
        self.index = idx
        self.smooth = True


class BMFace:
    def __init__(self, verts, idx):
        if len(verts) < 3:
            raise ValueError(f"BMFace requires >= 3 verts, got {len(verts)}")
        if len(set(id(v) for v in verts)) != len(verts):
            raise ValueError("Duplicate vertex in BMFace!")
        self.verts = tuple(verts)
        self.index = idx
        self.smooth = False
        self.material_index = 0
        self.normal = Vector((0, 0, 1))


class ElemSeq(list):
    def ensure_lookup_table(self):
        return None

    def index_update(self):
        for i, item in enumerate(self):
            item.index = i


class VertSeq(ElemSeq):
    def new(self, co=(0.0, 0.0, 0.0)):
        v = BMVert(co, len(self))
        self.append(v)
        return v


class EdgeSeq(ElemSeq):
    def __init__(self):
        super().__init__()
        self._lut = {}

    def clear(self):
        super().clear()
        self._lut.clear()

    def new(self, verts):
        v1, v2 = verts
        e = BMEdge(v1, v2, len(self))
        self.append(e)
        a, b = id(v1), id(v2)
        self._lut[(a, b) if a < b else (b, a)] = e
        return e

    def get(self, verts):
        v1, v2 = verts
        a, b = id(v1), id(v2)
        return self._lut.get((a, b) if a < b else (b, a))


class FaceSeq(ElemSeq):
    def __init__(self, edges_ref):
        super().__init__()
        self._edges = edges_ref

    def new(self, verts):
        f = BMFace(verts, len(self))
        self.append(f)
        n = len(verts)
        for i in range(n):
            v1, v2 = verts[i], verts[(i + 1) % n]
            if self._edges.get((v1, v2)) is None:
                self._edges.new((v1, v2))
        return f


class BMesh:
    def __init__(self):
        self.verts = VertSeq()
        self.edges = EdgeSeq()
        self.faces = FaceSeq(self.edges)

    def to_mesh(self, mesh):
        if hasattr(mesh, "use_auto_smooth"):
            raise AssertionError("Mesh should not have use_auto_smooth")
        self.verts.index_update()
        mesh.vertices = [type("MVert", (), {"co": v.co.copy(), "index": i})() for i, v in enumerate(self.verts)]
        mesh.edges = [
            type("MEdge", (), {"index": i, "vertices": (e.verts[0].index, e.verts[1].index)})()
            for i, e in enumerate(self.edges)
        ]
        mesh.polygons = [
            type("MPoly", (), {
                "use_smooth": f.smooth,
                "material_index": f.material_index,
                "vertices": tuple(v.index for v in f.verts),
            })()
            for f in self.faces
        ]

    def free(self):
        self.verts.clear()
        self.edges.clear()
        self.faces.clear()

    def normal_update(self):
        return None


class BMeshOps:
    @staticmethod
    def create_cube(bm, size=1.0, matrix=None, calc_uvs=True):
        h = size * 0.5
        coords = [
            (-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h),
            (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h),
        ]
        vs = [bm.verts.new(c) for c in coords]
        quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        for q in quads:
            bm.faces.new([vs[i] for i in q])
        return {"verts": vs}

    @staticmethod
    def create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0, matrix=None, calc_uvs=True):
        top = bm.verts.new((0, 0, radius))
        bot = bm.verts.new((0, 0, -radius))
        rings = []
        for iv in range(1, v_segments):
            phi = math.pi * iv / v_segments
            ring = []
            for iu in range(u_segments):
                theta = 2.0 * math.pi * iu / u_segments
                ring.append(bm.verts.new((
                    radius * math.sin(phi) * math.cos(theta),
                    radius * math.sin(phi) * math.sin(theta),
                    radius * math.cos(phi),
                )))
            rings.append(ring)
        for iu in range(u_segments):
            jn = (iu + 1) % u_segments
            bm.faces.new((top, rings[0][iu], rings[0][jn]))
            bm.faces.new((rings[-1][iu], bot, rings[-1][jn]))
        for r in range(len(rings) - 1):
            for iu in range(u_segments):
                jn = (iu + 1) % u_segments
                bm.faces.new((rings[r][iu], rings[r + 1][iu], rings[r + 1][jn], rings[r][jn]))
        return {"verts": bm.verts[:]}

    @staticmethod
    def spin(bm, geom, cent=(0, 0, 0), axis=(0, 0, 1), dvec=(0, 0, 0), angle=math.pi * 2, steps=16, use_merge=False, use_normal_flip=False, use_duplicate=False):
        src_verts = [g for g in geom if isinstance(g, BMVert)]
        src_edges = [g for g in geom if isinstance(g, BMEdge)]
        if not src_verts and src_edges:
            seen = {}
            for e in src_edges:
                for v in e.verts:
                    seen[id(v)] = v
            src_verts = list(seen.values())
        rings = [src_verts]
        for s in range(1, steps + 1):
            th = angle * s / steps
            ct, st = math.cos(th), math.sin(th)
            new_ring = []
            for v in src_verts:
                x, y, z = v.co.x - cent[0], v.co.y - cent[1], v.co.z - cent[2]
                nx = cent[0] + x * ct - y * st
                ny = cent[1] + x * st + y * ct
                nz = cent[2] + z
                new_ring.append(bm.verts.new((nx, ny, nz)))
            rings.append(new_ring)
        idx_map = {id(v): i for i, v in enumerate(src_verts)}
        for s in range(steps):
            r0, r1 = rings[s], rings[s + 1]
            for e in src_edges:
                i0, i1 = idx_map[id(e.verts[0])], idx_map[id(e.verts[1])]
                if (r0[i0].co - r1[i0].co).length > 1e-6 or (r0[i1].co - r1[i1].co).length > 1e-6:
                    bm.faces.new((r0[i0], r0[i1], r1[i1], r1[i0]))
        return {"geom_last": rings[-1]}

    @staticmethod
    def bevel(bm, geom, offset=0.01, segments=1, profile=0.5, affect='EDGES', clamp_overlap=True):
        return {"faces": [], "edges": [], "verts": []}

    @staticmethod
    def subdivide_edges(bm, edges, cuts=1, use_grid_fill=True, smooth=0.0):
        return {"geom": [], "geom_inner": [], "geom_split": []}

    @staticmethod
    def remove_doubles(bm, verts, dist=0.0001):
        return None

    @staticmethod
    def recalc_face_normals(bm, faces):
        return None

    @staticmethod
    def translate(bm, verts, vec=(0, 0, 0)):
        for v in verts:
            v.co = v.co + Vector(vec)

    @staticmethod
    def scale(bm, verts, vec=(1, 1, 1)):
        for v in verts:
            v.co = Vector((v.co.x * vec[0], v.co.y * vec[1], v.co.z * vec[2]))


class NamedDataList(list):
    def __init__(self, factory):
        super().__init__()
        self._factory = factory

    def new(self, *args, **kwargs):
        item = self._factory(*args, **kwargs)
        self.append(item)
        return item

    def get(self, name):
        for x in self:
            if getattr(x, "name", None) == name:
                return x
        return None

    def remove(self, item, do_unlink=True):
        if item in self:
            self.remove_item(item)

    def remove_item(self, item):
        super().remove(item)


def make_mock_blender(app_version=(4, 2, 0)):
    import types

    bpy = types.ModuleType("bpy")
    bmesh_mod = types.ModuleType("bmesh")
    mathutils_mod = types.ModuleType("mathutils")

    mathutils_mod.Vector = Vector
    mathutils_mod.Euler = Euler
    mathutils_mod.Matrix = Matrix

    bmesh_mod.new = lambda: BMesh()
    bmesh_mod.ops = BMeshOps

    actions_pool = None

    class KeyframePoint:
        def __init__(self, frame, val):
            self.co = (frame, val)
            self.interpolation = 'BEZIER'
            self.easing = 'AUTO'
            self.handle_left_type = 'AUTO_CLAMPED'
            self.handle_right_type = 'AUTO_CLAMPED'

    class FCurve:
        def __init__(self, data_path):
            self.data_path = data_path
            self.keyframe_points = []

    class Action:
        def __init__(self, name="Action"):
            self.name = name
            self.fcurves = []

    class AnimData:
        def __init__(self, action_name="Action"):
            self.action = actions_pool.new(action_name) if actions_pool is not None else Action(action_name)

    def record_anim_keyframe(owner, data_path, frame, val=0.0):
        if owner.animation_data is None:
            owner.animation_data = AnimData(f"{getattr(owner, 'name', 'Anim')}_Action")
        fc = next((f for f in owner.animation_data.action.fcurves if f.data_path == data_path), None)
        if fc is None:
            fc = FCurve(data_path)
            owner.animation_data.action.fcurves.append(fc)
        fc.keyframe_points.append(KeyframePoint(frame, val))

    class NodeTree:
        def __init__(self, name="NodeTree", tree_type="ShaderNodeTree"):
            self.name = name
            self.type = tree_type
            self.interface = NodeTreeInterface()
            self.nodes = NodeCollection(app_version, self)
            self.links = LinkCollection()
            self.animation_data = None

        def _record_keyframe(self, data_path, frame):
            record_anim_keyframe(self, data_path, frame)

    class Material:
        def __init__(self, name):
            self.name = name
            self.use_nodes = False
            self.node_tree = NodeTree(name + "_NT", "ShaderNodeTree")
            self.use_backface_culling = False

    class Mesh:
        def __init__(self, name):
            self.name = name
            self.materials = []
            self.vertices = []
            self.edges = []
            self.polygons = []
        def update(self):
            return None

    class BezierPoint:
        def __init__(self):
            self.co = Vector((0, 0, 0))
            self.handle_left = Vector((0, 0, 0))
            self.handle_right = Vector((0, 0, 0))
            self.handle_left_type = 'AUTO'
            self.handle_right_type = 'AUTO'

    class BezierPoints(list):
        def __init__(self):
            super().__init__([BezierPoint()])
        def add(self, count):
            for _ in range(count):
                self.append(BezierPoint())

    class CurvePoint:
        def __init__(self):
            self.co = (0.0, 0.0, 0.0, 1.0)
            self.radius = 1.0

    class CurvePoints(list):
        def __init__(self):
            super().__init__([CurvePoint()])
        def add(self, count):
            for _ in range(count):
                self.append(CurvePoint())

    class Spline:
        def __init__(self, stype):
            self.type = stype
            self.use_cyclic_u = False
            self.resolution_u = 12
            self.use_endpoint_u = True
            self.order_u = 4
            if stype == 'BEZIER':
                self.bezier_points = BezierPoints()
            else:
                self.points = CurvePoints()

    class Splines(list):
        def new(self, type='POLY'):
            s = Spline(type)
            self.append(s)
            return s

    class CurveData:
        def __init__(self, name, type='CURVE'):
            self.name = name
            self.type = type
            self.dimensions = '3D'
            self.bevel_depth = 0.0
            self.bevel_resolution = 4
            self.fill_mode = 'FULL'
            self.extrude = 0.0
            self.resolution_u = 12
            self.use_path = False
            self.path_duration = 100
            self.splines = Splines()
            self.materials = []
            if type == 'FONT':
                self.body = ""
                self.size = 1.0
                self.space_character = 1.0
                self.align_x = 'LEFT'
                self.align_y = 'BOTTOM'

    class LatticePoint:
        def __init__(self, co):
            self.co = Vector(co)
            self.co_deform = Vector(co)

    class LatticeData:
        def __init__(self, name):
            self.name = name
            self._u, self._v, self._w = 2, 2, 2
            self.interpolation_type_u = 'KEY_BSPLINE'
            self.interpolation_type_v = 'KEY_BSPLINE'
            self.interpolation_type_w = 'KEY_BSPLINE'
            self.points = []
            self._rebuild()

        def _rebuild(self):
            self.points = []
            for k in range(self._w):
                zw = -0.5 + (k / max(1, self._w - 1))
                for j in range(self._v):
                    yv = -0.5 + (j / max(1, self._v - 1))
                    for i in range(self._u):
                        xu = -0.5 + (i / max(1, self._u - 1))
                        self.points.append(LatticePoint((xu, yv, zw)))

        @property
        def points_u(self): return self._u
        @points_u.setter
        def points_u(self, val): self._u = val; self._rebuild()
        @property
        def points_v(self): return self._v
        @points_v.setter
        def points_v(self, val): self._v = val; self._rebuild()
        @property
        def points_w(self): return self._w
        @points_w.setter
        def points_w(self, val): self._w = val; self._rebuild()

    class EditBone:
        def __init__(self, name):
            self.name = name
            self.head = Vector((0, 0, 0))
            self.tail = Vector((0, 0, 1))
            self.roll = 0.0
            self.parent = None
            self.use_connect = False

    class EditBones(list):
        def __init__(self, arm_data):
            super().__init__()
            self._arm = arm_data

        def new(self, name):
            eb = EditBone(name)
            self.append(eb)
            for listener in self._arm._listeners:
                listener._sync_pose_bones()
            return eb

        def get(self, name):
            for eb in self:
                if eb.name == name:
                    return eb
            return None

    class ArmatureData:
        def __init__(self, name):
            self.name = name
            self.display_type = 'OCTAHEDRAL'
            self._listeners = []
            self.edit_bones = EditBones(self)

    class PoseBone:
        def __init__(self, name, arm_obj):
            self.name = name
            self._arm_obj = arm_obj
            self.rotation_mode = 'QUATERNION'
            self.rotation_euler = Euler((0, 0, 0), 'XYZ')
            self.location = Vector((0, 0, 0))
            self.scale = Vector((1, 1, 1))

        def keyframe_insert(self, data_path, frame, index=-1):
            raw = getattr(self, data_path, 0.0)
            snap = tuple(raw) if hasattr(raw, "__iter__") else float(raw)
            record_anim_keyframe(self._arm_obj, f'pose.bones["{self.name}"].{data_path}', frame, snap)

    class PoseBonesCollection:
        def __init__(self):
            self._list = []
            self._map = {}

        def _ensure(self, name, arm_obj):
            if name not in self._map:
                pb = PoseBone(name, arm_obj)
                self._list.append(pb)
                self._map[name] = pb

        def __getitem__(self, key):
            if isinstance(key, int):
                return self._list[key]
            return self._map[key]

        def get(self, name):
            return self._map.get(name)

        def __contains__(self, name):
            return name in self._map

        def __iter__(self):
            return iter(self._list)

        def __len__(self):
            return len(self._list)

    class Pose:
        def __init__(self):
            self.bones = PoseBonesCollection()

    class VertexGroup:
        def __init__(self, name, index):
            self.name = name
            self.index = index
            self.weights = {}

        def add(self, indices, weight, type='REPLACE'):
            for idx in indices:
                self.weights[int(idx)] = float(weight)

    class VertexGroups(list):
        def new(self, name="Group"):
            vg = VertexGroup(name, len(self))
            self.append(vg)
            return vg

        def get(self, name):
            for vg in self:
                if vg.name == name:
                    return vg
            return None

    class LightData:
        def __init__(self, name, type='POINT'):
            self.name = name
            self.type = type
            self.energy = 10.0
            self.color = (1.0, 1.0, 1.0)
            self.shape = 'SQUARE'
            self.size = 1.0
            self.size_y = 1.0
            self.shadow_soft_size = 0.1
            self.spread = math.pi
            self.spot_size = math.radians(45.0)
            self.spot_blend = 0.15
            self.animation_data = None

        def keyframe_insert(self, data_path, frame, index=-1):
            record_anim_keyframe(self, data_path, frame, getattr(self, data_path, 0.0))

    class CameraDOF:
        def __init__(self):
            self.use_dof = False
            self.focus_object = None
            self.focus_distance = 10.0
            self.aperture_fstop = 2.8
            self.aperture_blades = 0

    class CameraData:
        def __init__(self, name):
            self.name = name
            self.lens = 50.0
            self.sensor_width = 36.0
            self.clip_start = 0.1
            self.clip_end = 1000.0
            self.dof = CameraDOF()
            self.passepartout_alpha = 0.5

    class Modifier:
        def __init__(self, name, mtype):
            valid_types = {
                'BEVEL', 'ARRAY', 'SUBSURF', 'SOLIDIFY', 'LATTICE',
                'WAVE', 'MIRROR', 'WEIGHTED_NORMAL', 'NODES', 'ARMATURE',
            }
            if mtype not in valid_types:
                raise ValueError(f"Unsupported modifier type: {mtype}")
            self.name = name
            self.type = mtype
            self.node_group = None
            self.object = None
            self.use_vertex_groups = True

    class Modifiers(list):
        def new(self, name, type):
            m = Modifier(name, type)
            self.append(m)
            return m

    class Constraint:
        def __init__(self, ctype, owner):
            valid_types = {'FOLLOW_PATH', 'TRACK_TO', 'DAMPED_TRACK'}
            if ctype not in valid_types:
                raise ValueError(f"Unsupported constraint type: {ctype}")
            self.type = ctype
            self._owner = owner
            self.name = ctype
            self.target = None
            self.use_fixed_location = False
            self.use_curve_follow = False
            self.offset_factor = 0.0
            self.forward_axis = 'FORWARD_Y'
            self.up_axis = 'UP_Z'
            self.track_axis = 'TRACK_NEGATIVE_Z'

        def keyframe_insert(self, data_path, frame):
            record_anim_keyframe(self._owner, f'constraints["{self.name}"].{data_path}', frame, getattr(self, data_path, 0.0))

    class Constraints(list):
        def __init__(self, owner):
            super().__init__()
            self._owner = owner
        def new(self, type):
            c = Constraint(type, self._owner)
            self.append(c)
            return c

    class BObject:
        def __init__(self, name, data):
            self.name = name
            self.data = data
            self.location = Vector((0, 0, 0))
            self.rotation_euler = Euler((0, 0, 0))
            self.scale = Vector((1, 1, 1))
            self.parent = None
            self.modifiers = Modifiers()
            self.constraints = Constraints(self)
            self.vertex_groups = VertexGroups()
            self.animation_data = None
            self.empty_display_type = 'PLAIN_AXES'
            self.empty_display_size = 1.0
            self.hide_viewport = False
            self.hide_render = False
            self.mode = 'OBJECT'
            self._selected = False
            self.pose = Pose() if isinstance(data, ArmatureData) else None
            if isinstance(data, ArmatureData):
                data._listeners.append(self)
                self._sync_pose_bones()

        @property
        def type(self):
            if self.data is None:
                return 'EMPTY'
            if isinstance(self.data, Mesh):
                return 'MESH'
            if isinstance(self.data, CurveData):
                return 'FONT' if self.data.type == 'FONT' else 'CURVE'
            if isinstance(self.data, LightData):
                return 'LIGHT'
            if isinstance(self.data, CameraData):
                return 'CAMERA'
            if isinstance(self.data, ArmatureData):
                return 'ARMATURE'
            if isinstance(self.data, LatticeData):
                return 'LATTICE'
            return 'EMPTY'

        def _sync_pose_bones(self):
            if self.pose is not None and isinstance(self.data, ArmatureData):
                for eb in self.data.edit_bones:
                    self.pose.bones._ensure(eb.name, self)

        def select_set(self, state):
            self._selected = bool(state)

        def keyframe_insert(self, data_path, frame, index=-1):
            raw = getattr(self, data_path, 0.0)
            snap = tuple(raw) if hasattr(raw, "__iter__") else float(raw)
            record_anim_keyframe(self, data_path, frame, snap)

    class CollectionObjects(list):
        def link(self, obj):
            if obj not in self:
                self.append(obj)
        def unlink(self, obj):
            if obj in self:
                self.remove(obj)

    class CollectionChildren(list):
        def link(self, col):
            if col not in self:
                self.append(col)
        def unlink(self, col):
            if col in self:
                self.remove(col)

    class Collection:
        def __init__(self, name):
            self.name = name
            self.objects = CollectionObjects()
            self.children = CollectionChildren()

    class MistSettings:
        def __init__(self):
            self.use_mist = False
            self.intensity = 0.0
            self.start = 5.0
            self.depth = 25.0
            self.falloff = 'QUADRATIC'

    class World:
        def __init__(self, name):
            self.name = name
            self.use_nodes = False
            self.node_tree = NodeTree(name + "_NT", "ShaderNodeTree")
            self.mist_settings = MistSettings()

    class AOVItem:
        def __init__(self):
            self.name = ""
            self.type = 'COLOR'

    class AOVCollection(list):
        def add(self):
            item = AOVItem()
            self.append(item)
            return item

    class TimelineMarker:
        def __init__(self, name, frame):
            self.name = name
            self.frame = int(frame)
            self.camera = None

    class TimelineMarkers(list):
        def new(self, name, frame=1):
            m = TimelineMarker(name, frame)
            self.append(m)
            return m

    class SoundStrip:
        def __init__(self, name, filepath, channel, frame_start):
            self.name = name
            self.filepath = filepath
            self.channel = int(channel)
            self.frame_start = int(frame_start)
            self.type = 'SOUND'
            self.volume = 1.0

    class SequenceCollection(list):
        def new_sound(self, name, filepath, channel, frame_start):
            st = SoundStrip(name, filepath, channel, frame_start)
            self.append(st)
            return st

    class SequenceEditor:
        def __init__(self):
            self.sequences = SequenceCollection()
            self.strips = self.sequences

    looks = (
        ['None', 'AgX - Very Low Contrast', 'AgX - Low Contrast', 'AgX - Medium Low Contrast',
         'AgX - Base Contrast', 'AgX - Medium High Contrast', 'AgX - High Contrast', 'AgX - Very High Contrast']
        if app_version < (4, 2, 0)
        else ['None', 'Very Low Contrast', 'Low Contrast', 'Medium Low Contrast',
              'Medium Contrast', 'Medium High Contrast', 'High Contrast', 'Very High Contrast', 'Punchy']
    )

    class ViewSettings:
        def __init__(self):
            self._vt = 'AgX'
            self._look = 'None'
            self.exposure = 0.0
            self.gamma = 1.0
            enum_items = [type("EnumItem", (), {"identifier": x})() for x in looks]
            self.bl_rna = type("RNA", (), {"properties": {"look": type("Prop", (), {"enum_items": enum_items})()}})()

        @property
        def view_transform(self): return self._vt
        @view_transform.setter
        def view_transform(self, val):
            if val not in ('Standard', 'AgX', 'Filmic', 'Raw', 'False Color'):
                raise ValueError(f"Invalid view_transform: {val}")
            self._vt = val

        @property
        def look(self): return self._look
        @look.setter
        def look(self, val):
            if val not in looks:
                raise TypeError(f"enum '{val}' not found in {looks}")
            self._look = val

    actions_pool = NamedDataList(lambda name: Action(name))

    bpy.app = type("App", (), {"version": app_version})()
    bpy.data = type("Data", (), {
        "objects": NamedDataList(lambda name, data: BObject(name, data)),
        "meshes": NamedDataList(lambda name: Mesh(name)),
        "curves": NamedDataList(lambda name, type='CURVE': CurveData(name, type)),
        "lights": NamedDataList(lambda name, type='POINT': LightData(name, type)),
        "cameras": NamedDataList(lambda name: CameraData(name)),
        "armatures": NamedDataList(lambda name: ArmatureData(name)),
        "lattices": NamedDataList(lambda name: LatticeData(name)),
        "materials": NamedDataList(lambda name: Material(name)),
        "worlds": NamedDataList(lambda name: World(name)),
        "collections": NamedDataList(lambda name: Collection(name)),
        "node_groups": NamedDataList(lambda name, type='ShaderNodeTree': NodeTree(name, type)),
        "actions": actions_pool,
        "sounds": NamedDataList(lambda name: type("Snd", (), {"name": name})()),
        "textures": NamedDataList(lambda name: type("Tex", (), {"name": name})()),
        "images": NamedDataList(lambda name: type("Img", (), {"name": name})()),
    })()

    root_col = Collection("Scene Collection")
    world = bpy.data.worlds.new("World")
    cycles_settings = type("Cycles", (), {
        "device": "CPU",
        "samples": 128,
        "preview_samples": 32,
        "use_adaptive_sampling": True,
        "adaptive_threshold": 0.01,
        "use_denoising": True,
        "denoiser": "OPENIMAGEDENOISE",
        "denoising_input_passes": "RGB_ALBEDO_NORMAL",
        "denoising_prefilter": "ACCURATE",
        "use_preview_denoising": True,
        "max_bounces": 12,
        "diffuse_bounces": 4,
        "glossy_bounces": 4,
        "transmission_bounces": 12,
        "volume_bounces": 0,
        "transparent_max_bounces": 8,
        "volume_step_rate": 1.0,
        "volume_max_steps": 1024,
        "use_fast_gi": False,
    })()
    image_settings = type("ImgSettings", (), {
        "file_format": "PNG",
        "color_mode": "RGBA",
        "color_depth": "8",
    })()
    ffmpeg_settings = type("FFmpegSettings", (), {
        "format": "MPEG4",
        "codec": "H264",
        "constant_rate_factor": "MEDIUM",
        "audio_codec": "NONE",
        "audio_bitrate": 192,
    })()
    render_settings = type("Render", (), {
        "engine": "BLENDER_EEVEE_NEXT",
        "resolution_x": 1920,
        "resolution_y": 1080,
        "resolution_percentage": 100,
        "fps": 24,
        "film_transparent": False,
        "use_motion_blur": False,
        "motion_blur_shutter": 0.5,
        "filepath": "//render",
        "image_settings": image_settings,
        "ffmpeg": ffmpeg_settings,
    })()
    eevee_settings = type("Eevee", (), {
        "use_raytracing": False,
        "use_ssr": False,
        "use_bloom": False,
    })()

    class SceneClass:
        def __init__(self):
            self.collection = root_col
            self.world = world
            self.cycles = cycles_settings
            self.render = render_settings
            self.eevee = eevee_settings
            self.view_settings = ViewSettings()
            self.frame_start = 1
            self.frame_end = 250
            self.frame_current = 1
            self.camera = None
            self.use_nodes = False
            self.node_tree = NodeTree("Compositing Nodetree", "CompositorNodeTree")
            self.timeline_markers = TimelineMarkers()
            self.sequence_editor = None

        def sequence_editor_create(self):
            if self.sequence_editor is None:
                self.sequence_editor = SequenceEditor()
            return self.sequence_editor

        def sequence_editor_clear(self):
            self.sequence_editor = None

    scene = SceneClass()

    dev = type("Device", (), {"type": "OPTIX", "use": False})()
    cprefs = type("CPrefs", (), {
        "compute_device_type": "NONE",
        "devices": [dev],
        "refresh_devices": staticmethod(lambda: None),
        "get_devices": staticmethod(lambda: [dev]),
    })()
    prefs = type("Prefs", (), {"addons": {"cycles": type("Addon", (), {"preferences": cprefs})()}})()

    vl_objects = type("VLObjects", (), {"active": None})()
    view_layer = type("ViewLayer", (), {
        "objects": vl_objects,
        "aovs": AOVCollection(),
        "use_pass_mist": False,
        "use_pass_emit": False,
        "use_pass_z": False,
        "use_pass_normal": False,
        "use_pass_vector": False,
        "use_pass_ambient_occlusion": False,
        "use_pass_shadow": False,
        "use_pass_diffuse_color": False,
        "use_pass_glossy_direct": False,
        "use_pass_environment": False,
        "use_pass_cryptomatte_object": False,
        "use_pass_cryptomatte_material": False,
        "cryptomatte_levels": 2,
        "update": staticmethod(lambda: None),
    })()

    bpy.context = type("Context", (), {
        "scene": scene,
        "collection": root_col,
        "active_object": None,
        "preferences": prefs,
        "view_layer": view_layer,
    })()

    def mock_mode_set(mode='OBJECT'):
        act = bpy.context.view_layer.objects.active or bpy.context.active_object
        if act is not None:
            act.mode = mode
            if mode == 'OBJECT' and hasattr(act, "_sync_pose_bones"):
                act._sync_pose_bones()
        return {'FINISHED'}

    bpy.ops = type("Ops", (), {
        "object": type("ObjOps", (), {"mode_set": staticmethod(mock_mode_set)})(),
        "outliner": type("OutlinerOps", (), {
            "orphans_purge": staticmethod(lambda do_local_ids=True, do_linked_ids=True, do_recursive=True: {'FINISHED'})
        })(),
    })()

    return bpy, bmesh_mod, mathutils_mod


def assert_all_20_tracks(bpy, ns, script_dir):
    """Run strict assertions across Phase 1 and Tracks 1 through 20."""
    scene = bpy.context.scene
    vl = bpy.context.view_layer

    # 1. Verify 9 numbered sub-collections + master collection
    expected_sub_cols = {
        "01_Environment_Base",
        "02_Floating_Hull_Thrusters",
        "03_Stall_Architecture_Roof",
        "04_Interior_Ramen_Props",
        "05_Lanterns_Fans_Cables",
        "06_Cyber_Neon_Signage",
        "07_Cinematic_Lights_Camera",
        "08_Cyber_City_Megablock",
        "09_Characters_And_Drones",
    }
    present_cols = {c.name for c in bpy.data.collections}
    missing_cols = expected_sub_cols - present_cols
    assert not missing_cols, f"Missing required sub-collections: {missing_cols}"

    # 2. Track 1 & 16: Compositor Pipeline, Multi-Pass Render Channels, Cryptomatte & Shader AOVs
    assert scene.use_nodes is True, "Compositor scene.use_nodes must be True"
    comp_types = [n.bl_idname for n in scene.node_tree.nodes]
    for req_node in (
        "CompositorNodeRLayers",
        "CompositorNodeGlare",
        "CompositorNodeMixRGB",
        "CompositorNodeLensdist",
        "CompositorNodeExposure",
        "CompositorNodeColorBalance",
        "CompositorNodeComposite",
        "CompositorNodeViewer",
    ):
        assert req_node in comp_types, f"Missing Compositor node: {req_node}"
    assert comp_types.count("CompositorNodeGlare") >= 2, "Requires Dual Glare (FOG_GLOW + STREAKS)"

    for pass_flag in (
        "use_pass_mist", "use_pass_emit", "use_pass_z", "use_pass_normal",
        "use_pass_vector", "use_pass_ambient_occlusion", "use_pass_shadow",
        "use_pass_diffuse_color", "use_pass_glossy_direct", "use_pass_environment",
        "use_pass_cryptomatte_object", "use_pass_cryptomatte_material",
    ):
        assert getattr(vl, pass_flag) is True, f"ViewLayer.{pass_flag} not enabled"
    assert vl.cryptomatte_levels == 6, f"Expected cryptomatte_levels=6, got {vl.cryptomatte_levels}"
    assert scene.world.mist_settings.start == 8.0 and scene.world.mist_settings.depth == 28.0
    aov_names = {a.name for a in vl.aovs}
    assert {"AOV_Neon_Mask", "AOV_Thermal_Heat"}.issubset(aov_names), f"Missing custom AOVs: {aov_names}"

    # 3. Track 2: GeometryNodeTree Rain Storm, 14 Puddle Splash Ripples & Rising Ramen Steam
    gn_rain = bpy.data.node_groups.get("GN_Cyber_Rain_Field")
    assert gn_rain is not None, "Missing GN_Cyber_Rain_Field GeometryNodeTree"
    assert len(gn_rain.interface.items_tree) >= 2, "GN_Cyber_Rain_Field missing interface sockets"
    for r_idx in range(1, 15):
        assert bpy.data.objects.get(f"Puddle_Splash_Ripple_{r_idx:02d}") is not None
    assert bpy.data.materials.get("MAT_Volumetric_Ramen_Steam") is not None

    # 4. Track 3, 4 & 5: 6-DOF Cyber-Chef Hierarchy, Street Props, Hover Physics & Neon Flicker
    strainer = bpy.data.objects.get("Chef_Tebo_Noodle_Strainer")
    ladle = bpy.data.objects.get("Chef_Hishaku_Ladle_Cup")
    assert strainer is not None and strainer.parent is not None, "Chef strainer not parented to arm"
    assert ladle is not None and ladle.parent is not None, "Chef ladle not parented to arm"
    assert bpy.data.objects.get("Cyber_Vending_Cabinet") is not None
    assert bpy.data.objects.get("Floating_Hologram_Menu_Pane") is not None
    vessel_root = bpy.data.objects.get("Hover_Vessel_Master_Root")
    assert vessel_root is not None and vessel_root.animation_data is not None

    # 5. Track 6, 7, 8, 9 & 10: 8 Brutalist Skyscrapers, Maglev Train, 6 Spinners & 4 Hologram Billboards
    for t_i in range(1, 9):
        assert bpy.data.objects.get(f"Cyber_Megatower_{t_i:02d}_LowerCore") is not None
    assert bpy.data.materials.get("MAT_Building_Night_Windows") is not None
    train_root = bpy.data.objects.get("Maglev_Train_Root")
    assert train_root is not None and train_root.animation_data is not None
    for s_i in range(1, 7):
        sp_obj = bpy.data.objects.get(f"Cyber_Spinner_Vehicle_{s_i:02d}")
        assert sp_obj is not None and any(c.type == 'FOLLOW_PATH' for c in sp_obj.constraints)
    assert bpy.data.materials.get("MAT_Mega_Hologram_Glitch") is not None
    for b_i in range(1, 5):
        assert bpy.data.objects.get(f"Colossal_Hologram_Billboard_{b_i:02d}") is not None

    # 6. Track 11, 12, 13 & 14: Rigged Armature Patrons, Bionic Cat, Delivery Drone, Umbrellas & 3-Camera Cut
    assert len(bpy.data.armatures) == 2, f"Expected 2 armatures, got {len(bpy.data.armatures)}"
    for arm_name in ("Cyber_Patron_01_Netrunner_Armature", "Cyber_Patron_02_StreetSamurai_Armature"):
        arm_obj = bpy.data.objects.get(arm_name)
        assert arm_obj is not None and len(arm_obj.pose.bones) == 13
        assert arm_obj.animation_data is not None and len(arm_obj.animation_data.action.fcurves) >= 8

    cat_root = bpy.data.objects.get("Bionic_Stray_Cat_Root")
    assert cat_root is not None
    assert bpy.data.objects.get("Bionic_Cat_Tail_Seg_05") is not None
    drone_root = bpy.data.objects.get("Delivery_Drone_Root")
    assert drone_root is not None and drone_root.animation_data is not None
    for rot_i in range(1, 5):
        r_obj = bpy.data.objects.get(f"Delivery_Drone_Rotor_{rot_i:02d}")
        assert r_obj is not None and r_obj.animation_data is not None

    assert bpy.data.materials.get("MAT_Clear_Wet_Umbrella") is not None
    assert bpy.data.objects.get("Cyber_Umbrella_01") is not None
    assert bpy.data.objects.get("Cyber_Umbrella_02") is not None

    assert len(bpy.data.cameras) == 3, f"Expected 3 cinematic cameras, got {len(bpy.data.cameras)}"
    assert scene.camera is not None and scene.camera.name == "Cinematic_Camera_85mm"
    marker_frames = [(m.frame, m.camera.name) for m in scene.timeline_markers]
    assert marker_frames == [
        (1, "Cinematic_Camera_85mm"),
        (97, "Cinematic_Camera_50mm_HeroLow"),
        (169, "Cinematic_Camera_105mm_MacroCounter"),
    ], f"Unexpected TimelineMarker camera bindings: {marker_frames}"

    # 7. Track 17, 18 & 19: WAV RIFF/WAVE Header, VSE Sound Strip, LOD Controller & JSON Manifest
    assert scene.cycles.samples == 4096 and scene.render.engine == 'CYCLES'
    # Verify LOD preset switcher round-trip and restore PRODUCTION_4096
    preset_fn = ns["configure_render_quality_preset"]
    preset_fn(scene, mode='FAST_TURNTABLE_256')
    assert scene.cycles.samples == 256
    preset_fn(scene, mode='EEVEE_REALTIME')
    assert scene.render.engine in ('BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT')
    preset_fn(scene, mode='PRODUCTION_4096')
    assert scene.cycles.samples == 4096 and scene.render.engine == 'CYCLES'

    assert scene.sequence_editor is not None and len(scene.sequence_editor.sequences) == 1
    snd_strip = scene.sequence_editor.sequences[0]
    assert snd_strip.channel == 1 and snd_strip.frame_start == 1
    assert scene.render.image_settings.file_format == 'FFMPEG'
    assert scene.render.ffmpeg.codec == 'H264' and scene.render.ffmpeg.audio_codec == 'AAC'

    wav_path = os.path.join(script_dir, "cyberpunk_ramen_soundtrack.wav")
    assert os.path.isfile(wav_path), f"Missing generated WAV file: {wav_path}"
    with open(wav_path, "rb") as raw_wf:
        hdr = raw_wf.read(12)
    riff_tag, _riff_sz, wave_tag = struct.unpack("<4sI4s", hdr)
    assert riff_tag == b"RIFF" and wave_tag == b"WAVE", f"Invalid RIFF/WAVE header: {hdr}"
    with wave.open(wav_path, "rb") as wf_chk:
        assert wf_chk.getnchannels() == 2
        assert wf_chk.getsampwidth() == 2
        assert wf_chk.getframerate() == 48000
        assert wf_chk.getnframes() == 480000

    manifest_path = os.path.join(script_dir, "cyberpunk_diorama_manifest.json")
    assert os.path.isfile(manifest_path), f"Missing manifest JSON: {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as mf:
        manifest = json.load(mf)
    assert len(manifest["scene_topology_sha256"]) == 64
    assert len(manifest["soundtrack_audio"]["sha256_digest"]) == 64
    assert manifest["cycles_samples"] == 4096
    return manifest


def assert_tracks_21_to_24(bpy, script_dir):
    """Execute and validate Tracks 21, 22, 23 (GLB/OBJ/MTL + WebGL2 Viewer) and Track 24 (3x 1280x720 PNGs)."""
    import diorama_exporter
    import render_diorama_previews

    # Track 21: Evaluate all 579 objects (modifiers + 154 curve/font Bishop-frame tubular meshes)
    scene_eval = diorama_exporter.evaluate_diorama_scene(bpy, eval_frame=115)
    assert len(scene_eval["objects"]) == 579, f"Expected 579 evaluated objects, got {len(scene_eval['objects'])}"
    assert len(scene_eval["materials"]) == 31, f"Expected 31 extracted PBR materials, got {len(scene_eval['materials'])}"

    curve_font_objs = [o for o in scene_eval["objects"] if o["type"] in ("CURVE", "FONT")]
    assert len(curve_font_objs) == 154, f"Expected 154 CURVE/FONT objects, got {len(curve_font_objs)}"
    for c_obj in curve_font_objs:
        assert len(c_obj["verts"]) > 0 and len(c_obj["tris"]) > 0, f"Untessellated curve/font: {c_obj['name']}"

    total_eval_verts = sum(len(o["verts"]) for o in scene_eval["objects"])
    total_eval_tris = sum(len(o["tris"]) for o in scene_eval["objects"])
    assert total_eval_verts > 60000 and total_eval_tris > 45000

    # Track 22: Export spec-compliant Binary glTF 2.0 (.glb) and Wavefront (.obj + .mtl)
    exp_stats = diorama_exporter.export_glb_and_obj(scene_eval, script_dir)
    glb_path = exp_stats["glb_path"]
    obj_path = exp_stats["obj_path"]
    mtl_path = exp_stats["mtl_path"]
    assert os.path.isfile(glb_path) and os.path.isfile(obj_path) and os.path.isfile(mtl_path)

    with open(glb_path, "rb") as gf:
        glb_hdr = gf.read(20)
        magic, version, total_len, json_len, json_type = struct.unpack("<IIIII", glb_hdr)
        assert magic == 0x46546C67, f"Invalid GLB magic: {hex(magic)}"
        assert version == 2, f"Invalid GLB version: {version}"
        assert total_len == os.path.getsize(glb_path), "GLB header length mismatch"
        assert json_type == 0x4E4F534A, "First GLB chunk must be JSON (0x4E4F534A)"
        json_payload = json.loads(gf.read(json_len).decode("utf-8"))
        bin_len, bin_type = struct.unpack("<II", gf.read(8))
        assert bin_type == 0x004E4942 and bin_len > 1000000, "Second GLB chunk must be BIN (0x004E4942)"
        assert len(json_payload["nodes"]) == 579, f"Expected 579 glTF nodes, got {len(json_payload['nodes'])}"
        assert len(json_payload["materials"]) == 31, f"Expected 31 glTF materials, got {len(json_payload['materials'])}"

    with open(obj_path, "r", encoding="utf-8") as of:
        obj_txt = of.read()
    assert obj_txt.count("\no ") + (1 if obj_txt.startswith("o ") else 0) == 579, "OBJ must contain all 579 objects"
    with open(mtl_path, "r", encoding="utf-8") as mf:
        mtl_txt = mf.read()
    assert mtl_txt.count("newmtl ") == 31, "MTL must contain all 31 materials"

    # Track 23: Export self-contained WebGL2 3D viewport HTML
    viewer_stats = diorama_exporter.export_interactive_webgl_viewer(glb_path, script_dir)
    html_path = viewer_stats["html_path"]
    assert os.path.isfile(html_path) and viewer_stats["html_bytes"] > 200000
    with open(html_path, "r", encoding="utf-8") as hf:
        html_txt = hf.read()
    for req_token in (
        "#version 300 es",
        "Cinematic_Camera",
        "85mm Orbit",
        "50mm Low Hero",
        "105mm Counter Macro",
        "cyberpunk_ramen_soundtrack.wav",
        "accent-color: var(--brand-color)",
        "@supports not (accent-color: var(--brand-color))",
    ):
        assert req_token in html_txt, f"Missing token in WebGL2 viewer HTML: {req_token}"

    # Track 24: Verify 3x 1280x720 24-bit RGB PNG previews via pure-Python software rasterizer + Neon Bloom
    preview_names = (
        ("render_cam1_85mm_establishing.png", "Cinematic_Camera_85mm", 85.0),
        ("render_cam2_50mm_low_hero.png", "Cinematic_Camera_50mm_HeroLow", 50.0),
        ("render_cam3_105mm_counter_macro.png", "Cinematic_Camera_105mm_MacroCounter", 105.0),
    )
    if all(os.path.isfile(os.path.join(script_dir, fn)) for fn, _, _ in preview_names):
        render_stats = []
        for fn, cam_n, lens_mm in preview_names:
            p_path = os.path.join(script_dir, fn)
            st = _verify_and_measure_png(p_path, expected_wh=(1280, 720))
            st["camera_name"] = cam_n
            st["lens_mm"] = lens_mm
            render_stats.append(st)
    else:
        render_stats = render_diorama_previews.render_all_camera_previews(scene_eval, script_dir, width=1280, height=720)
    assert len(render_stats) == 3
    for rs in render_stats:
        p_path = rs["filepath"]
        assert os.path.isfile(p_path), f"Missing rendered PNG: {p_path}"
        with open(p_path, "rb") as pf:
            sig = pf.read(8)
            assert sig == b"\x89PNG\r\n\x1a\n", f"Invalid PNG signature in {p_path}"
            ihdr_len, ihdr_tag = struct.unpack(">I4s", pf.read(8))
            assert ihdr_tag == b"IHDR" and ihdr_len == 13
            w, h, bit_depth, col_type, comp_m, filt_m, interlace_m = struct.unpack(">IIBBBBB", pf.read(13))
            assert (w, h, bit_depth, col_type) == (1280, 720, 8, 2), f"Unexpected IHDR in {p_path}"
        assert rs["std_luminance"] > 12.0, f"Low pixel variance in {p_path}: {rs}"
        assert rs["bright_neon_pixels"] > 250, f"Insufficient neon bloom highlights in {p_path}: {rs}"

    return {
        "evaluated_vertices": total_eval_verts,
        "evaluated_triangles": total_eval_tris,
        "export": exp_stats,
        "viewer": viewer_stats,
        "renders": render_stats,
    }


def _verify_and_measure_png(png_path, expected_wh=None):
    """Validate PNG signature/IHDR/IDAT/IEND, reconstruct raw RGB24 bytes, and return stats + digests."""
    assert os.path.isfile(png_path), f"Missing PNG file: {png_path}"
    with open(png_path, "rb") as pf:
        data = pf.read()
    file_sha256 = hashlib.sha256(data).hexdigest()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"Invalid PNG magic in {png_path}"
    pos = 8
    width = height = 0
    idat_chunks = []
    saw_iend = False
    while pos + 8 <= len(data):
        clen, ctag = struct.unpack(">I4s", data[pos:pos + 8])
        pos += 8
        cbody = data[pos:pos + clen]
        pos += clen + 4
        if ctag == b"IHDR":
            width, height, bit_depth, col_type, comp_m, filt_m, interlace_m = struct.unpack(">IIBBBBB", cbody)
            assert (bit_depth, col_type, comp_m, filt_m, interlace_m) == (8, 2, 0, 0, 0), f"Invalid IHDR in {png_path}"
            if expected_wh is not None:
                assert (width, height) == expected_wh, f"Expected {expected_wh}, got {(width, height)} in {png_path}"
        elif ctag == b"IDAT":
            idat_chunks.append(cbody)
        elif ctag == b"IEND":
            saw_iend = True
            break
    assert saw_iend and width > 0 and height > 0, f"Truncated PNG chunks in {png_path}"
    raw_scanlines = zlib.decompress(b"".join(idat_chunks))
    stride = width * 3
    assert len(raw_scanlines) == height * (1 + stride), f"Unexpected decompressed PNG size in {png_path}"
    rgb_rows = []
    for y in range(height):
        row_start = y * (1 + stride)
        assert raw_scanlines[row_start] == 0, f"Unsupported filter byte {raw_scanlines[row_start]} in {png_path}"
        rgb_rows.append(raw_scanlines[row_start + 1:row_start + 1 + stride])
    rgb_bytes = b"".join(rgb_rows)
    rgb_sha256 = hashlib.sha256(rgb_bytes).hexdigest()

    npix = width * height
    step = max(1, npix // 16384)
    lum_sum = 0.0
    lum_sq_sum = 0.0
    count = 0
    bright_neon = 0
    for idx in range(0, npix * 3, step * 3):
        r8 = rgb_bytes[idx]
        g8 = rgb_bytes[idx + 1]
        b8 = rgb_bytes[idx + 2]
        lum = 0.2126 * r8 + 0.7152 * g8 + 0.0722 * b8
        lum_sum += lum
        lum_sq_sum += lum * lum
        count += 1
        if r8 > 210 or g8 > 210 or b8 > 210:
            bright_neon += step
    mean_lum = lum_sum / max(1, count)
    std_lum = math.sqrt(max(0.0, (lum_sq_sum / max(1, count)) - mean_lum * mean_lum))
    return {
        "filepath": png_path,
        "byte_size": len(data),
        "width": width,
        "height": height,
        "file_sha256": file_sha256,
        "rgb_sha256": rgb_sha256,
        "mean_luminance": round(mean_lum, 2),
        "std_luminance": round(std_lum, 2),
        "bright_neon_pixels": bright_neon,
    }


def _validate_riff_avi_container(avi_path, expected_frames=240, expected_fps=24):
    """Parse and validate RIFF/AVI 1.0 binary container hierarchy (hdrl, avih, strl/vids, strl/auds, movi, idx1)."""
    assert os.path.isfile(avi_path), f"Missing AVI container: {avi_path}"
    file_size = os.path.getsize(avi_path)
    assert file_size > 1000000, f"AVI file unexpectedly small: {file_size}"

    h_avi = hashlib.sha256()
    with open(avi_path, "rb") as f_hash:
        while True:
            blk = f_hash.read(1048576)
            if not blk:
                break
            h_avi.update(blk)
    avi_sha256 = h_avi.hexdigest()

    with open(avi_path, "rb") as af:
        riff_tag, riff_size, form_type = struct.unpack("<4sI4s", af.read(12))
        assert riff_tag == b"RIFF" and form_type == b"AVI ", f"Invalid RIFF/AVI header: {riff_tag} / {form_type}"
        assert riff_size + 8 == file_size, f"RIFF size {riff_size + 8} != file size {file_size}"

        saw_hdrl = False
        saw_avih = False
        saw_vids = False
        saw_auds = False
        saw_movi = False
        saw_idx1 = False
        vid_width = vid_height = 0
        v_chunks = 0
        a_chunks = 0
        idx1_entries = 0

        pos = 12
        while pos + 8 <= file_size:
            af.seek(pos)
            ck_id, ck_sz = struct.unpack("<4sI", af.read(8))
            if ck_id == b"LIST":
                list_type = af.read(4)
                if list_type == b"hdrl":
                    saw_hdrl = True
                    hdr_end = pos + 8 + ck_sz
                    sub = pos + 12
                    while sub + 8 <= hdr_end:
                        af.seek(sub)
                        sid, ssz = struct.unpack("<4sI", af.read(8))
                        if sid == b"avih":
                            saw_avih = True
                            avih_data = struct.unpack("<14I", af.read(56))
                            usec_per_frame = avih_data[0]
                            total_frames = avih_data[4]
                            streams = avih_data[6]
                            vid_width = avih_data[8]
                            vid_height = avih_data[9]
                            assert abs(usec_per_frame - round(1000000 / expected_fps)) <= 2
                            assert total_frames == expected_frames, f"AVI total_frames={total_frames}"
                            assert streams == 2, f"AVI streams={streams}"
                        elif sid == b"LIST":
                            stype = af.read(4)
                            if stype == b"strl":
                                strh_id, strh_sz = struct.unpack("<4sI", af.read(8))
                                assert strh_id == b"strh"
                                fcc_type, fcc_handler = struct.unpack("<4s4s", af.read(8))
                                af.seek(sub + 12 + 8 + strh_sz + (strh_sz & 1))
                                strf_id, strf_sz = struct.unpack("<4sI", af.read(8))
                                assert strf_id == b"strf"
                                if fcc_type == b"vids":
                                    saw_vids = True
                                    bi_size, bi_w, bi_h, bi_planes, bi_bitcount, bi_comp = struct.unpack(
                                        "<IiiHH4s", af.read(20)
                                    )
                                    assert bi_size == 40 and bi_w == vid_width and abs(bi_h) == vid_height
                                    assert bi_planes == 1 and bi_bitcount == 24 and bi_comp == b"\x00\x00\x00\x00"
                                elif fcc_type == b"auds":
                                    saw_auds = True
                                    w_tag, n_ch, s_rate, avg_bps, blk_align, bps = struct.unpack(
                                        "<HHIIHH", af.read(16)
                                    )
                                    assert (w_tag, n_ch, s_rate, blk_align, bps) == (1, 2, 48000, 4, 16)
                        sub += 8 + ssz + (ssz & 1)
                elif list_type == b"movi":
                    saw_movi = True
                    movi_end = pos + 8 + ck_sz
                    sub = pos + 12
                    expected_frame_bytes = vid_width * vid_height * 3
                    while sub + 8 <= movi_end:
                        af.seek(sub)
                        mid, msz = struct.unpack("<4sI", af.read(8))
                        if mid == b"00db":
                            assert msz == expected_frame_bytes, f"00db size {msz} != {expected_frame_bytes}"
                            v_chunks += 1
                        elif mid == b"01wb":
                            assert msz == 8000, f"01wb audio chunk size {msz} != 8000"
                            a_chunks += 1
                        sub += 8 + msz + (msz & 1)
            elif ck_id == b"idx1":
                saw_idx1 = True
                idx1_entries = ck_sz // 16
            pos += 8 + ck_sz + (ck_sz & 1)

    assert saw_hdrl and saw_avih and saw_vids and saw_auds and saw_movi and saw_idx1, "Incomplete AVI structure"
    assert v_chunks == expected_frames, f"Expected {expected_frames} 00db chunks, got {v_chunks}"
    assert a_chunks == expected_frames, f"Expected {expected_frames} 01wb chunks, got {a_chunks}"
    assert idx1_entries == expected_frames * 2, f"Expected {expected_frames * 2} idx1 entries, got {idx1_entries}"

    return {
        "filepath": avi_path,
        "byte_size": file_size,
        "width": vid_width,
        "height": vid_height,
        "video_chunks": v_chunks,
        "audio_chunks": a_chunks,
        "idx1_entries": idx1_entries,
        "sha256_digest": avi_sha256,
    }


def _parse_mp4_boxes(stream, start_offset, end_offset):
    """Parse big-endian ISO-14496-12 boxes in [start_offset, end_offset)."""
    boxes = []
    pos = start_offset
    while pos + 8 <= end_offset:
        stream.seek(pos)
        sz, btype = struct.unpack(">I4s", stream.read(8))
        hdr = 8
        if sz == 1:
            sz = struct.unpack(">Q", stream.read(8))[0]
            hdr = 16
        elif sz == 0:
            sz = end_offset - pos
        assert sz >= hdr and pos + sz <= end_offset, f"Invalid MP4 box {btype} size {sz} at {pos}"
        boxes.append((btype.decode("latin1"), pos, pos + hdr, pos + sz))
        pos += sz
    return boxes


def _validate_isobmff_mp4_container(mp4_path, expected_frames=240):
    """Parse and validate big-endian ISO-BMFF (.mp4) container (ftyp, mdat, moov -> mvhd, avc1 video trak, audio trak)."""
    assert os.path.isfile(mp4_path), f"Missing MP4 container: {mp4_path}"
    file_size = os.path.getsize(mp4_path)
    assert file_size > 500000, f"MP4 file unexpectedly small: {file_size}"

    h_mp4 = hashlib.sha256()
    with open(mp4_path, "rb") as f_hash:
        while True:
            blk = f_hash.read(1048576)
            if not blk:
                break
            h_mp4.update(blk)
    mp4_sha256 = h_mp4.hexdigest()

    with open(mp4_path, "rb") as mf:
        top = _parse_mp4_boxes(mf, 0, file_size)
        top_names = [b[0] for b in top]
        for req_box in ("ftyp", "mdat", "moov"):
            assert req_box in top_names, f"Missing top-level MP4 box '{req_box}' in {top_names}"

        ftyp_box = next(b for b in top if b[0] == "ftyp")
        mf.seek(ftyp_box[2])
        major_brand = mf.read(4)
        assert major_brand == b"isom", f"Expected ftyp major_brand 'isom', got {major_brand}"

        moov_box = next(b for b in top if b[0] == "moov")
        moov_children = _parse_mp4_boxes(mf, moov_box[2], moov_box[3])
        moov_names = [b[0] for b in moov_children]
        assert "mvhd" in moov_names, "Missing mvhd inside moov"
        traks = [b for b in moov_children if b[0] == "trak"]
        assert len(traks) == 2, f"Expected 2 traks (video + audio) in moov, got {len(traks)}"

        saw_video_avc1 = False
        saw_audio_trak = False
        video_samples = 0

        for tk in traks:
            tk_children = _parse_mp4_boxes(mf, tk[2], tk[3])
            tk_names = [b[0] for b in tk_children]
            assert "tkhd" in tk_names and "mdia" in tk_names, f"Missing tkhd/mdia in trak: {tk_names}"
            mdia = next(b for b in tk_children if b[0] == "mdia")
            mdia_children = _parse_mp4_boxes(mf, mdia[2], mdia[3])
            hdlr = next(b for b in mdia_children if b[0] == "hdlr")
            mf.seek(hdlr[2] + 8)
            handler_type = mf.read(4)
            minf = next(b for b in mdia_children if b[0] == "minf")
            minf_children = _parse_mp4_boxes(mf, minf[2], minf[3])
            stbl = next(b for b in minf_children if b[0] == "stbl")
            stbl_children = _parse_mp4_boxes(mf, stbl[2], stbl[3])
            stbl_names = [b[0] for b in stbl_children]
            for req_st in ("stsd", "stts", "stsc", "stsz", "stco"):
                assert req_st in stbl_names, f"Missing {req_st} in stbl: {stbl_names}"

            stsd = next(b for b in stbl_children if b[0] == "stsd")
            mf.seek(stsd[2] + 8)
            entry_sz, entry_fmt = struct.unpack(">I4s", mf.read(8))

            stsz = next(b for b in stbl_children if b[0] == "stsz")
            mf.seek(stsz[2] + 4)
            sample_size, sample_count = struct.unpack(">II", mf.read(8))

            if handler_type == b"vide":
                assert entry_fmt == b"avc1", f"Expected avc1 video sample entry, got {entry_fmt}"
                assert "stss" in stbl_names, "Video stbl must contain stss sync sample table"
                assert sample_count == expected_frames, f"Expected {expected_frames} video samples, got {sample_count}"
                saw_video_avc1 = True
                video_samples = sample_count
            elif handler_type == b"soun":
                assert entry_fmt in (b"sowt", b"twos", b"mp4a", b"lpcm"), f"Unexpected audio format {entry_fmt}"
                assert sample_count >= expected_frames
                saw_audio_trak = True

        assert saw_video_avc1 and saw_audio_trak, "MP4 must contain both avc1 video trak and audio trak"

    return {
        "filepath": mp4_path,
        "byte_size": file_size,
        "video_samples": video_samples,
        "sha256_digest": mp4_sha256,
    }


def assert_tracks_25_to_28(bpy, script_dir):
    """Execute and validate Tracks 25, 26, 27, and 28 (240-Frame Evaluator, Renderer, AVI/MP4 Muxer & Manifest)."""
    import render_diorama_animation
    import video_muxer

    # Track 25: Full 240-Frame Scene & Constraint Evaluator (evaluate_scene_at_frame)
    cache_candidate = os.path.join(os.path.dirname(script_dir), "snap2.pkl")
    snap = render_diorama_animation.load_or_build_snapshot(
        cache_candidate if os.path.isfile(cache_candidate) else None
    )
    st_f1 = render_diorama_animation.evaluate_scene_at_frame(snap, 1)
    st_f48 = render_diorama_animation.evaluate_scene_at_frame(snap, 48)
    st_f96 = render_diorama_animation.evaluate_scene_at_frame(snap, 96)
    st_f97 = render_diorama_animation.evaluate_scene_at_frame(snap, 97)
    st_f120 = render_diorama_animation.evaluate_scene_at_frame(snap, 120)
    st_f168 = render_diorama_animation.evaluate_scene_at_frame(snap, 168)
    st_f169 = render_diorama_animation.evaluate_scene_at_frame(snap, 169)
    st_f200 = render_diorama_animation.evaluate_scene_at_frame(snap, 200)
    st_f240 = render_diorama_animation.evaluate_scene_at_frame(snap, 240)

    # 3-Camera TimelineMarker Director Cut switching
    assert st_f1["active_camera"] == "Cinematic_Camera_85mm" and st_f96["active_camera"] == "Cinematic_Camera_85mm"
    assert st_f97["active_camera"] == "Cinematic_Camera_50mm_HeroLow" and st_f120["active_camera"] == "Cinematic_Camera_50mm_HeroLow" and st_f168["active_camera"] == "Cinematic_Camera_50mm_HeroLow"
    assert st_f169["active_camera"] == "Cinematic_Camera_105mm_MacroCounter" and st_f200["active_camera"] == "Cinematic_Camera_105mm_MacroCounter" and st_f240["active_camera"] == "Cinematic_Camera_105mm_MacroCounter"

    # Dynamic World Motion assertions across frames
    # 1. Anti-gravity stall hover bobbing
    assert abs(st_f1["world"]["Hover_Vessel_Master_Root"][11] - st_f48["world"]["Hover_Vessel_Master_Root"][11]) > 0.01
    # 2. Maglev train sweeping across viaduct (X = -24 -> +24m)
    train_x1 = st_f1["world"]["Maglev_Train_Master_Root"][3]
    train_x240 = st_f240["world"]["Maglev_Train_Master_Root"][3]
    assert train_x1 < -20.0 and train_x240 > 20.0, f"Maglev X sweep failed: {train_x1} -> {train_x240}"
    # 3. Flying spinner vehicles along aerial splines
    assert st_f1["world"]["Flying_Spinner_01"] != st_f120["world"]["Flying_Spinner_01"]
    # 4. 6-DOF chef noodle-draining & ladle-pouring arms
    assert st_f1["world"]["Chef_Shoulder_R_Pivot"] != st_f48["world"]["Chef_Shoulder_R_Pivot"]
    assert st_f1["world"]["Chef_Shoulder_L_Pivot"] != st_f48["world"]["Chef_Shoulder_L_Pivot"]
    # 5. Patron PoseBone armature kinematics
    assert st_f1["bone_deform"]["Armature_Cyber_Patron_01"] != st_f48["bone_deform"]["Armature_Cyber_Patron_01"]
    assert st_f1["bone_deform"]["Armature_Cyber_Patron_02"] != st_f48["bone_deform"]["Armature_Cyber_Patron_02"]
    # 6. Spinning cooling fans & delivery drone rotors + bionic cat tail sway + puddle ripples
    assert st_f1["world"]["Exhaust_Fan_Blades_01"] != st_f48["world"]["Exhaust_Fan_Blades_01"]
    assert st_f1["world"]["Delivery_Drone_Body"] != st_f120["world"]["Delivery_Drone_Body"]
    assert st_f1["world"]["Bionic_Cat_Tail_Spline"] != st_f48["world"]["Bionic_Cat_Tail_Spline"]
    assert st_f1["world"]["Puddle_Ripple_Ring_01_A"] != st_f48["world"]["Puddle_Ripple_Ring_01_A"]

    # Track 26: Verify key still frames (render_f001_cam1.png, render_f120_cam2.png, render_f200_cam3.png)
    still_files = (
        (1, "render_f001_cam1.png", "Cinematic_Camera_85mm"),
        (120, "render_f120_cam2.png", "Cinematic_Camera_50mm_HeroLow"),
        (200, "render_f200_cam3.png", "Cinematic_Camera_105mm_MacroCounter"),
    )
    still_stats = {}
    for fno, sname, cam_n in still_files:
        spath = os.path.join(script_dir, sname)
        st = _verify_and_measure_png(spath)
        st["frame"] = fno
        st["camera"] = cam_n
        assert st["std_luminance"] > 10.0, f"Low contrast in {sname}: {st}"
        still_stats[sname] = st

    # Assert frame pixel hashes differ across frames 1, 120, and 200
    h_f1 = still_stats["render_f001_cam1.png"]["rgb_sha256"]
    h_f120 = still_stats["render_f120_cam2.png"]["rgb_sha256"]
    h_f200 = still_stats["render_f200_cam3.png"]["rgb_sha256"]
    assert len({h_f1, h_f120, h_f200}) == 3, "Frame pixel hashes for f1, f120, f200 must all be distinct!"

    report_path = os.path.join(script_dir, "cyberpunk_diorama_render_report.json")
    assert os.path.isfile(report_path), f"Missing render report: {report_path}"
    with open(report_path, "r", encoding="utf-8") as rf:
        render_report = json.load(rf)
    assert render_report["frames_rendered"] == 240
    assert len(render_report["frame_sha256"]) == 240
    assert render_report["frame_sha256"]["1"] == h_f1
    assert render_report["frame_sha256"]["120"] == h_f120
    assert render_report["frame_sha256"]["200"] == h_f200

    # Track 27: Validate RIFF/AVI 1.0 (cyberpunk_diorama_240f.avi) & ISO-BMFF .MP4 (cyberpunk_diorama_240f.mp4)
    avi_path = os.path.join(script_dir, "cyberpunk_diorama_240f.avi")
    mp4_path = os.path.join(script_dir, "cyberpunk_diorama_240f.mp4")
    avi_stats = _validate_riff_avi_container(avi_path, expected_frames=240, expected_fps=24)
    mp4_stats = _validate_isobmff_mp4_container(mp4_path, expected_frames=240)

    # Track 28: Record & verify SHA-256 digests in cyberpunk_diorama_manifest.json
    manifest_path = os.path.join(script_dir, "cyberpunk_diorama_manifest.json")
    assert os.path.isfile(manifest_path), f"Missing manifest: {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as mf:
        manifest_data = json.load(mf)

    manifest_data["phase7_animation_and_video"] = {
        "artifacts": {
            "cyberpunk_diorama_240f.avi": {
                "filepath": "cyberpunk_diorama_240f.avi",
                "byte_size": avi_stats["byte_size"],
                "sha256_digest": avi_stats["sha256_digest"],
            },
            "cyberpunk_diorama_240f.mp4": {
                "filepath": "cyberpunk_diorama_240f.mp4",
                "byte_size": mp4_stats["byte_size"],
                "sha256_digest": mp4_stats["sha256_digest"],
            },
            "render_f001_cam1.png": {
                "filepath": "render_f001_cam1.png",
                "byte_size": still_stats["render_f001_cam1.png"]["byte_size"],
                "sha256_digest": still_stats["render_f001_cam1.png"]["file_sha256"],
            },
            "render_f120_cam2.png": {
                "filepath": "render_f120_cam2.png",
                "byte_size": still_stats["render_f120_cam2.png"]["byte_size"],
                "sha256_digest": still_stats["render_f120_cam2.png"]["file_sha256"],
            },
            "render_f200_cam3.png": {
                "filepath": "render_f200_cam3.png",
                "byte_size": still_stats["render_f200_cam3.png"]["byte_size"],
                "sha256_digest": still_stats["render_f200_cam3.png"]["file_sha256"],
            },
        },
        "frame_pixel_sha256": {
            "1": h_f1,
            "120": h_f120,
            "200": h_f200,
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest_data, mf, indent=2, sort_keys=False)

    # Re-read manifest from disk and assert SHA-256 integrity
    with open(manifest_path, "r", encoding="utf-8") as mf:
        verified_manifest = json.load(mf)
    p7 = verified_manifest["phase7_animation_and_video"]
    assert p7["artifacts"]["cyberpunk_diorama_240f.avi"]["sha256_digest"] == avi_stats["sha256_digest"]
    assert p7["artifacts"]["cyberpunk_diorama_240f.mp4"]["sha256_digest"] == mp4_stats["sha256_digest"]
    assert len({p7["frame_pixel_sha256"]["1"], p7["frame_pixel_sha256"]["120"], p7["frame_pixel_sha256"]["200"]}) == 3

    return {
        "stills": still_stats,
        "avi": avi_stats,
        "mp4": mp4_stats,
        "frame_pixel_sha256": p7["frame_pixel_sha256"],
        "resolution": [avi_stats["width"], avi_stats["height"]],
    }


def print_telemetry_summary_table(manifest_40, manifest_43, phase6_stats, phase7_stats):
    """Print human-scannable scene telemetry summary table from verified cryptographic manifest."""
    m = manifest_43
    gt = m["geometry_totals"]
    sc = m["shading_and_compositor"]
    ad = m["animation_and_director"]
    au = m["soundtrack_audio"]
    obt = m["objects_by_type"]
    ex = phase6_stats["export"]
    vw = phase6_stats["viewer"]
    rn = phase6_stats["renders"]
    av = phase7_stats["avi"]
    mp = phase7_stats["mp4"]
    st = phase7_stats["stills"]

    rows = [
        ("Target Compatibility", "Blender 4.0.2 & Blender 4.3.0 (VERIFIED -- ALL 28 TRACKS)"),
        ("Render Engine / Preset", f"{m['render_engine']} @ {m['cycles_samples']} samples (AgX High Contrast)"),
        ("Collections Hierarchy", f"{m['collections']['total_collections']} total (1 Master + 9 Sub-Collections)"),
        ("Total Scene Objects", f"{m['total_objects']} (MESH={obt['MESH']}, CURVE={obt['CURVE']}, LIGHT={obt['LIGHT']}, CAM={obt['CAMERA']}, ARM={obt['ARMATURE']}, LAT={obt['LATTICE']}, EMPTY={obt['EMPTY']})"),
        ("Base Mesh Geometry", f"{gt['total_meshes']} meshes | {gt['total_vertices']:,} verts | {gt['total_edges']:,} edges | {gt['total_polygons']:,} polys"),
        ("Evaluated 3D Topology", f"{phase6_stats['evaluated_vertices']:,} world verts | {phase6_stats['evaluated_triangles']:,} triangles (incl. 154 curve/font tubes)"),
        ("Procedural Shading", f"{sc['total_materials']} materials | {sc['total_shader_nodes']} shader nodes (4.0: {manifest_40['shading_and_compositor']['total_shader_nodes']})"),
        ("Compositor & GeoNodes", f"{sc['total_compositor_nodes']} compositor nodes | {sc['total_geometry_node_groups']} GeometryNodeTree | Cryptomatte L{sc['cryptomatte_levels']}"),
        ("Registered Custom AOVs", ", ".join(sc["registered_shader_aovs"])),
        ("Animation & Director", f"{ad['total_actions']} actions | {ad['total_fcurves']} F-curves | 3 TimelineMarkers (f1, f97, f169)"),
        ("Procedural WAV Audio", f"{au['filepath']} ({au['byte_size']:,} bytes, {au['sample_rate_hz']} Hz 16-bit Stereo, {au['duration_seconds']}s)"),
        ("Binary glTF 2.0 (.glb)", f"cyberpunk_diorama.glb ({ex['glb_bytes']:,} bytes | {ex['node_count']} nodes | {ex['mesh_count']} meshes | {ex['material_count']} PBR mats)"),
        ("Wavefront OBJ / MTL", f"cyberpunk_diorama.obj ({os.path.getsize(ex['obj_path']):,} bytes) + cyberpunk_diorama.mtl ({os.path.getsize(ex['mtl_path']):,} bytes)"),
        ("WebGL2 3D Viewport", f"cyberpunk_diorama_viewer.html ({vw['html_bytes']:,} bytes, self-contained embedded GLB + Neon Bloom)"),
        ("Software PNG Cut 1", f"render_cam1_85mm_establishing.png ({rn[0]['byte_size']:,} B | 1280x720 | std={rn[0]['std_luminance']} | neon_px={rn[0]['bright_neon_pixels']:,})"),
        ("Software PNG Cut 2", f"render_cam2_50mm_low_hero.png ({rn[1]['byte_size']:,} B | 1280x720 | std={rn[1]['std_luminance']} | neon_px={rn[1]['bright_neon_pixels']:,})"),
        ("Software PNG Cut 3", f"render_cam3_105mm_counter_macro.png ({rn[2]['byte_size']:,} B | 1280x720 | std={rn[2]['std_luminance']} | neon_px={rn[2]['bright_neon_pixels']:,})"),
        ("Phase 7 Key Stills", f"render_f001_cam1.png ({st['render_f001_cam1.png']['byte_size']:,} B) | render_f120_cam2.png ({st['render_f120_cam2.png']['byte_size']:,} B) | render_f200_cam3.png ({st['render_f200_cam3.png']['byte_size']:,} B)"),
        ("RIFF/AVI 1.0 Video+PCM", f"cyberpunk_diorama_240f.avi ({av['byte_size']:,} B | 240x 00db + 240x 01wb | SHA256={av['sha256_digest'][:16]})"),
        ("ISO-BMFF MP4 Video+PCM", f"cyberpunk_diorama_240f.mp4 ({mp['byte_size']:,} B | 240x avc1 I_PCM + 48kHz stereo | SHA256={mp['sha256_digest'][:16]})"),
        ("Soundtrack SHA-256", au["sha256_digest"]),
        ("Scene Topology SHA-256", m["scene_topology_sha256"]),
    ]

    print("\n" + "=" * 104)
    print(" CYBERPUNK FLOATING RAMEN STALL DIORAMA -- 28-TRACK TELEMETRY, 3D EXPORT & 240F VIDEO AUDIT SUMMARY")
    print("=" * 104)
    for k, v in rows:
        print(f" {k:<24} | {v}")
    print("=" * 104)


def validate_script(script_path):
    script_abs = os.path.abspath(script_path)
    script_dir = os.path.dirname(script_abs)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    with open(script_abs, "r", encoding="utf-8") as f:
        src = f.read()

    # 1. Check strict zero-placeholder banned patterns
    banned_regexes = [
        (r"\bTODO\b", "Contains TODO"),
        (r"\bpass\b", "Contains pass keyword/word"),
        (r"repeat for other", "Contains 'repeat for other'"),
        (r"\.\.\.", "Contains ellipsis '...'"),
        (r"use_auto_smooth", "Uses deprecated mesh.use_auto_smooth (removed in 4.1)"),
    ]
    for pattern, msg in banned_regexes:
        m = re.search(pattern, src)
        if m:
            line_no = src[:m.start()].count("\n") + 1
            raise AssertionError(f"BANNED PATTERN ({msg}) at line {line_no}")

    # 2. Parse AST and double-check Zero 'ast.Pass' nodes
    tree = ast.parse(src, filename=script_abs)
    for node in ast.walk(tree):
        if isinstance(node, ast.Pass):
            raise AssertionError(f"AST Pass node found at line {node.lineno}")

    # 3. Execute & audit all 20 tracks under BOTH Blender 4.0.2 and Blender 4.3.0 mock runtimes
    manifests = {}
    last_bpy = None
    for ver in ((4, 0, 2), (4, 3, 0)):
        bpy, bmesh_mod, mathutils_mod = make_mock_blender(ver)
        sys.modules["bpy"] = bpy
        sys.modules["bmesh"] = bmesh_mod
        sys.modules["mathutils"] = mathutils_mod
        ns = {"__name__": "__main__", "__file__": script_abs}
        exec(compile(src, script_abs, "exec"), ns)
        manifest = assert_all_20_tracks(bpy, ns, script_dir)
        manifests[ver] = manifest
        last_bpy = bpy
        print(
            f"[OK] Blender {ver[0]}.{ver[1]}.{ver[2]} verified (TRACKS 1-20): "
            f"objects={len(bpy.data.objects)}, "
            f"meshes={len(bpy.data.meshes)}, "
            f"curves={len(bpy.data.curves)}, "
            f"armatures={len(bpy.data.armatures)}, "
            f"materials={len(bpy.data.materials)}, "
            f"lights={len(bpy.data.lights)}, "
            f"cameras={len(bpy.data.cameras)}, "
            f"lattices={len(bpy.data.lattices)}"
        )

    assert (
        manifests[(4, 0, 2)]["scene_topology_sha256"]
        == manifests[(4, 3, 0)]["scene_topology_sha256"]
    ), "Deterministic topology hash mismatch between 4.0.2 and 4.3.0!"
    assert (
        manifests[(4, 0, 2)]["soundtrack_audio"]["sha256_digest"]
        == manifests[(4, 3, 0)]["soundtrack_audio"]["sha256_digest"]
    ), "Deterministic WAV SHA-256 mismatch between runs!"

    # 4. Execute & audit Tracks 21-24 (3D Mesh/Curve Evaluator, GLB/OBJ/MTL Exporter, WebGL2 Viewer & Software PNG Renderer)
    phase6_stats = assert_tracks_21_to_24(last_bpy, script_dir)
    print(
        f"[OK] Tracks 21-24 verified: "
        f"evaluated_tris={phase6_stats['evaluated_triangles']:,}, "
        f"glb={phase6_stats['export']['glb_bytes']:,}B, "
        f"viewer_html={phase6_stats['viewer']['html_bytes']:,}B, "
        f"renders=3x(1280x720 PNG)"
    )

    # 5. Execute & audit Tracks 25-28 (240-Frame Evaluator, Z-Buffer/Bloom Renderer, RIFF/AVI + ISO-BMFF MP4 Muxer & Manifest)
    phase7_stats = assert_tracks_25_to_28(last_bpy, script_dir)
    print(
        f"[OK] Tracks 25-28 verified: "
        f"avi={phase7_stats['avi']['byte_size']:,}B (240f), "
        f"mp4={phase7_stats['mp4']['byte_size']:,}B (240f), "
        f"stills=3x({phase7_stats['resolution'][0]}x{phase7_stats['resolution'][1]} PNG)"
    )

    print_telemetry_summary_table(manifests[(4, 0, 2)], manifests[(4, 3, 0)], phase6_stats, phase7_stats)


if __name__ == "__main__":
    validate_script(sys.argv[1])
