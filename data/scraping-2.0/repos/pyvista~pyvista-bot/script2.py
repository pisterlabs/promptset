from vtk_info import get_cls_info
import openai
from key import OPENAI_KEY

openai.api_key = OPENAI_KEY


def get_src(vtk_info_in, vtk_info_out, temperature=1.0, n=1):
    gpt_prompt = f'''
Input:
{vtk_info_in['cls_name']}
{vtk_info_in['fnames']}

Output:
import vtk as _vtk
import pyvista
from pyvista.utilities import check_valid_vector
import numpy as np

def CircularArcFromNormal(center, resolution=100, normal=None, polar=None, angle=None):
    check_valid_vector(center, 'center')
    if normal is None:
        normal = [0, 0, 1]
    if polar is None:
        polar = [1, 0, 0]
    if angle is None:
        angle = 90.0

    arc = _vtk.vtkArcSource()
    arc.SetCenter(*center)
    arc.SetResolution(resolution)
    arc.UseNormalAndAngleOn()
    check_valid_vector(normal, 'normal')
    arc.SetNormal(*normal)
    check_valid_vector(polar, 'polar')
    arc.SetPolarVector(*polar)
    arc.SetAngle(angle)
    arc.Update()
    angle = np.deg2rad(arc.GetAngle())
    arc = pyvista.wrap(arc.GetOutput())
    # Compute distance of every point along circular arc
    center = np.array(center)
    radius = np.sqrt(np.sum((arc.points[0] - center) ** 2, axis=0))
    angles = np.linspace(0.0, angle, resolution + 1)
    arc['Distance'] = radius * angles
    return arc


def test_circular_arc_from_normal():
    center = [0, 0, 0]
    normal = [0, 0, 1]
    polar = [-2.0, 0, 0]
    angle = 90
    resolution = 100

    mesh = CircularArcFromNormal(center, resolution, normal, polar, angle)
    assert mesh.n_points == resolution + 1
    assert mesh.n_cells == 1
    distance = np.arange(0.0, 1.0 + 0.01, 0.01) * np.pi
    assert np.allclose(mesh['Distance'], distance)


Input:
{vtk_info_out['cls_name']}
{vtk_info_out['fnames']}

Output:
'''

    src_response = openai.Completion.create(
        engine="code-davinci-002",
        prompt=gpt_prompt,
        temperature=temperature,
        max_tokens=1024,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop='Input:',
        n=n,
    )

    return [choice['text'] for choice in src_response['choices']]


def get_docstr(vtk_info_in, vtk_info_out, func_sig, verbose=False):
    gpt_prompt = f'''
Input:
{vtk_info_in['cls_name']}
{vtk_info_in['short_desc']}
{vtk_info_in['long_desc']}
{vtk_info_in['fnames']}

def strip(
    self,
    join=False,
    max_length=1000,
    pass_cell_data=False,
    pass_cell_ids=False,
    pass_point_ids=False,
    progress_bar=False,
):

Output:
    """Strip poly data cells.

    Generates triangle strips and/or poly-lines from input
    polygons, triangle strips, and lines.

    Polygons are assembled into triangle strips only if they are
    triangles; other types of polygons are passed through to the
    output and not stripped. (Use ``triangulate`` filter to
    triangulate non-triangular polygons prior to running this
    filter if you need to strip all the data.) The filter will
    pass through (to the output) vertices if they are present in
    the input polydata.

    Also note that if triangle strips or polylines are defined in
    the input they are passed through and not joined nor
    extended. (If you wish to strip these use ``triangulate``
    filter to fragment the input into triangles and lines prior to
    running this filter.)

    This filter implements `vtkStripper
    <https://vtk.org/doc/nightly/html/classvtkStripper.html>`_

    Parameters
    ----------
    join : bool, optional
        If ``True``, the output polygonal segments will be joined
        if they are contiguous. This is useful after slicing a
        surface. The default is ``False``.

    max_length : int, optional
        Specify the maximum number of triangles in a triangle
        strip, and/or the maximum number of lines in a poly-line.

    pass_cell_data : bool, optional
        Enable/Disable passing of the CellData in the input to the
        output as FieldData. Note the field data is transformed.
        Default is ``False``.

    pass_cell_ids : bool, optional
        If ``True``, the output polygonal dataset will have a
        celldata array that holds the cell index of the original
        3D cell that produced each output cell. This is useful for
        picking. The default is ``False`` to conserve memory.

    pass_point_ids : bool, optional
        If ``True``, the output polygonal dataset will have a
        pointdata array that holds the point index of the original
        vertex that produced each output vertex. This is useful
        for picking. The default is ``False`` to conserve memory.

    progress_bar : bool, optional
        Display a progress bar to indicate progress.

    Returns
    -------
    pyvista.PolyData
        Stripped mesh.
    """

Input:
{vtk_info_out['cls_name']}
{vtk_info_out['short_desc']}
{vtk_info_out['long_desc']}
{vtk_info_out['fnames']}

{func_sig}

Output:

'''

    if verbose:
        print(gpt_prompt)
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=gpt_prompt,
        temperature=0.5,
        max_tokens=1024,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop='Input:',
    )

    text = response['choices'][0]['text']
    if verbose:
        print(text)
    return text

print('Acquiring VTK class definitions...')
vtk_info_in = get_cls_info('vtkArcSource')
vtk_info_out = get_cls_info('vtkEllipseArcSource')

print('Querying OpenAI...')
# docstr = get_docstr(vtk_info_in, vtk_info_out, verbose=True)
sources = get_src(vtk_info_in, vtk_info_out, temperature=0.5, n=2)

for src in sources:
    print(src)


