import os
from paraview.simple import *

# Set input and output directories
input_dir = "fenics2"  # Replace with your input directory
output_dir = "xdmf_to_vtk"  # Replace with your output directory

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over all files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".xdmf"):
        input_path = os.path.join(input_dir, filename)
        
        # Extract base name without extension for output naming
        base_name = os.path.splitext(filename)[0]
        
        # Load the XDMF file
        reader = Xdmf3ReaderS(registrationName=filename, FileName=[input_path])
        reader.PointArrays = ['uh']

        # Get animation scene and update based on data timesteps
        animation_scene = GetAnimationScene()
        animation_scene.UpdateAnimationUsingDataTimeSteps()

        # Get active view and show data
        render_view = GetActiveViewOrCreate('RenderView')
        display = Show(reader, render_view, 'UnstructuredGridRepresentation')

        # Configure display properties
        display.Representation = 'Surface'
        display.ColorArrayName = [None, '']
        display.OSPRayScaleArray = 'uh'
        display.OSPRayScaleFunction = 'PiecewiseFunction'
        display.SelectOrientationVectors = 'uh'
        display.ScaleFactor = 0.1
        display.SetScaleArray = ['POINTS', 'uh']
        display.ScaleTransferFunction = 'PiecewiseFunction'
        display.OpacityArray = ['POINTS', 'uh']
        display.OpacityTransferFunction = 'PiecewiseFunction'
        display.OpacityArrayName = ['POINTS', 'uh']

        # Update view and set scalar coloring
        render_view.Update()
        ColorBy(display, ('POINTS', 'uh', 'Magnitude'))
        display.RescaleTransferFunctionToDataRange(True, False)
        display.SetScalarBarVisibility(render_view, True)

        # Save data in the output directory
        output_path = os.path.join(output_dir, f"sample_{base_name}_time.vtk")
        SaveData(output_path, proxy=reader, PointDataArrays=['uh'],
                 Writetimestepsasfileseries=1,
                 FileType='Ascii')

print("Processing complete. Files saved in the output directory.")

