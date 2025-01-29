import h5py
import numpy as np
import vtk
from vtk.util import numpy_support
import os
import glob
import re

def vtk_to_h5(vtk_dir, output_h5_file):
    # Create a new HDF5 file
    with h5py.File(output_h5_file, 'w') as h5f:
        # Find all vtk files in the directory
        vtk_files = sorted(glob.glob(os.path.join(vtk_dir, 'output_sample_elasticity_2d_e*_time_*.vtk')))
        #vtk_files = sorted(glob.glob(os.path.join(vtk_dir, 'output_sample_heat_2d_hc*_time_*.vtk')))
        print(vtk_files)
        sample_time_data = {}

        for vtk_file in vtk_files:
            # Extract sample index and time index from the filename
            match = re.match(r"output_sample_elasticity_2d_e(\d+)_time_(\d+)\.vtk", os.path.basename(vtk_file))
            #match = re.match(r".*sample_heat_2d_hc(\d+)_time_(\d+)\.vtk", os.path.basename(vtk_file))
            if not match:
                continue
            sample_idx, time_idx = match.groups()
            sample_idx = int(sample_idx)
            time_idx = int(time_idx)
            print('match =',match)

            # Read the VTK file using vtkStructuredGridReader
            reader = vtk.vtkStructuredGridReader()
            reader.SetFileName(vtk_file)
            reader.Update()
            vtk_data = reader.GetOutput()

            if not vtk_data or vtk_data.GetNumberOfPoints() == 0:
                print(f"Warning: No data found in file {vtk_file}")
                continue

            # Get grid dimensions
            dims = vtk_data.GetDimensions()
            xdim, ydim = dims[0], dims[1]

            # Extract the grid points
            points = vtk_data.GetPoints()
            coords = numpy_support.vtk_to_numpy(points.GetData())
            x_coords = np.unique(coords[:, 0])
            y_coords = np.unique(coords[:, 1])

            # Extract vector data arrays
            num_arrays = vtk_data.GetPointData().GetNumberOfArrays()
            vectors = []
            for i in range(num_arrays):
                vtk_array = vtk_data.GetPointData().GetArray(i)
                if vtk_array:
                    numpy_array = numpy_support.vtk_to_numpy(vtk_array)
                    numpy_array = numpy_array.reshape((xdim, ydim), order='C')
                    vectors.append(numpy_array)

            if not vectors:
                print(f"Warning: No arrays found in file {vtk_file}")
                continue

            vectors = np.stack(vectors, axis=-1)  # Shape: (xdim, ydim, v)

            # Store the data in a dictionary
            if sample_idx not in sample_time_data:
                sample_time_data[sample_idx] = []
            sample_time_data[sample_idx].append((time_idx, vectors))

            # Save the grid points only once
            #if 'grid' not in h5f:
            #    h5f.create_dataset('grid/x', data=x_coords, dtype='float32')
            #    h5f.create_dataset('grid/y', data=y_coords, dtype='float32')

        # Write the data to the HDF5 file
        for sample_idx, time_data in sample_time_data.items():
            # Sort by time index
            time_data.sort(key=lambda x: x[0])
            time_steps = len(time_data)
            vectors_shape = time_data[0][1].shape  # (xdim, ydim, v)

            # Create datasets for each sample
            group = h5f.create_group(f"{sample_idx}")
            data_shape = (time_steps, vectors_shape[0], vectors_shape[1], vectors_shape[2])
            data_dset = group.create_dataset("data", shape=data_shape, dtype='float32')

            # Populate the data array and time grid
            t_grid = []
            for idx, (time_idx, vectors) in enumerate(time_data):
                data_dset[idx] = vectors
                t_grid.append(time_idx)

            # Create the grid group with x, y, and t arrays
            grid_group = group.create_group("grid")
            grid_group.create_dataset("x", data=x_coords, dtype='float32')
            grid_group.create_dataset("y", data=y_coords, dtype='float32')
            grid_group.create_dataset("t", data=np.array(t_grid, dtype='float32'))

            print(f"Sample {sample_idx} written with {time_steps} time steps.")

def vtk_to_h5_2(vtk_dir, output_h5_file):
    # Create a new HDF5 file
    with h5py.File(output_h5_file, 'w') as h5f:
        # Find all vtk files in the directory
        vtk_files = sorted(glob.glob(os.path.join(vtk_dir, 'output_sample_elasticity_2d_e*_time_*.vtk')))
        print(vtk_files)
        sample_time_data = {}

        for vtk_file in vtk_files:
            # Extract sample index and time index from the filename
            match = re.match(r"output_sample_elasticity_2d_e(\d+)_time_(\d+)\.vtk", os.path.basename(vtk_file))
            if not match:
                continue
            sample_idx, time_idx = match.groups()
            sample_idx = int(sample_idx)
            time_idx = int(time_idx)

            # Read the VTK file using vtkStructuredGridReader
            reader = vtk.vtkStructuredGridReader()
            reader.SetFileName(vtk_file)
            reader.Update()
            vtk_data = reader.GetOutput()

            if not vtk_data or vtk_data.GetNumberOfPoints() == 0:
                print(f"Warning: No data found in file {vtk_file}")
                continue

            # Get grid dimensions
            dims = vtk_data.GetDimensions()
            xdim, ydim = dims[0], dims[1]

            # Extract the grid points
            points = vtk_data.GetPoints()
            coords = numpy_support.vtk_to_numpy(points.GetData())
            x_coords = np.unique(coords[:, 0])
            y_coords = np.unique(coords[:, 1])

            # Extract vector data arrays
            num_arrays = vtk_data.GetPointData().GetNumberOfArrays()
            vectors = []
            for i in range(num_arrays):
                vtk_array = vtk_data.GetPointData().GetArray(i)
                if vtk_array:
                    numpy_array = numpy_support.vtk_to_numpy(vtk_array)

                    # Handle multi-field data (e.g., elasticity or heat conduction)
                    num_points = xdim * ydim
                    num_components = vtk_array.GetNumberOfComponents()
                    if numpy_array.size == num_points * num_components:
                        numpy_array = numpy_array.reshape((xdim, ydim, num_components), order='C')
                        # MODIFICATION: Keep only the first two components if available
                        if num_components >= 2:
                            numpy_array = numpy_array[:, :, :2]
                    else:
                        raise ValueError(
                            f"Unexpected array size: {numpy_array.size} does not match expected size "
                            f"{num_points} * {num_components} = {num_points * num_components}"
                        )
                    vectors.append(numpy_array)

            if not vectors:
                print(f"Warning: No arrays found in file {vtk_file}")
                continue

            # Combine all arrays into one, with shape: (xdim, ydim, num_components)
            vectors = np.stack(vectors, axis=-1).squeeze()  # Removes redundant dimensions if `v=1`

            # Store the data in a dictionary
            if sample_idx not in sample_time_data:
                sample_time_data[sample_idx] = []
            sample_time_data[sample_idx].append((time_idx, vectors))

        # Write the data to the HDF5 file
        for sample_idx, time_data in sample_time_data.items():
            # Sort by time index
            time_data.sort(key=lambda x: x[0])
            time_steps = len(time_data)
            xdim, ydim, num_components = time_data[0][1].shape  # Dimensions of the data

            # Create datasets for each sample
            group = h5f.create_group(f"{sample_idx}")
            data_shape = (time_steps, xdim, ydim, num_components)  # Desired shape
            data_dset = group.create_dataset("data", shape=data_shape, dtype='float32')

            # Populate the data array and time grid
            t_grid = []
            for idx, (time_idx, vectors) in enumerate(time_data):
                data_dset[idx] = vectors
                t_grid.append(time_idx)

            # Create the grid group with x, y, and t arrays
            grid_group = group.create_group("grid")
            grid_group.create_dataset("x", data=x_coords, dtype='float32')
            grid_group.create_dataset("y", data=y_coords, dtype='float32')
            grid_group.create_dataset("t", data=np.array(t_grid, dtype='float32'))

            print(f"Sample {sample_idx} written with {time_steps} time steps.")


# ... (remaining functions h5_to_vtk, convert_unstructured_to_structured, process_directory, printHDF5contents remain unchanged)

def convert_unstructured_to_structured(input_file, output_file):
    # Load the unstructured grid data
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(input_file)
    reader.Update()

    unstructured_grid = reader.GetOutput()

    # Get points from the unstructured grid
    points = unstructured_grid.GetPoints()
    num_points = points.GetNumberOfPoints()

    # Convert VTK points to a NumPy array for easier manipulation
    points_array = np.array([points.GetPoint(i) for i in range(num_points)])

    # Print the bounds of the points for verification
    print(f"Processing file: {input_file}")
    print(f"Bounds of points: {unstructured_grid.GetBounds()}")
    print(f"Total number of points: {num_points}")

    # Determine the unique x and y coordinates
    x_coords = np.unique(points_array[:, 0])
    y_coords = np.unique(points_array[:, 1])

    # Sort the coordinates to ensure proper ordering
    x_coords.sort()
    y_coords.sort()

    # Get the dimensions of the grid
    Nx = len(x_coords)
    Ny = len(y_coords)
    print(f"Determined grid dimensions: Nx = {Nx}, Ny = {Ny}")

    # Create a structured grid and set its dimensions
    structured_grid = vtk.vtkStructuredGrid()
    structured_grid.SetDimensions(Nx, Ny, 1)  # 2D grid, depth = 1

    # Create a new ordered points array matching structured grid order (row-major order)
    ordered_points = []
    point_indices = []

    for y in y_coords:
        for x in x_coords:
            # Find the index of the point that matches (x, y)
            index = np.where(np.isclose(points_array[:, 0], x) & np.isclose(points_array[:, 1], y))[0][0]
            ordered_points.append(points_array[index])
            point_indices.append(index)

    # Convert the ordered points to a vtkPoints object
    vtk_ordered_points = vtk.vtkPoints()
    for point in ordered_points:
        vtk_ordered_points.InsertNextPoint(point)

    structured_grid.SetPoints(vtk_ordered_points)

    # Reorder point data (if available) to match the new point order
    if unstructured_grid.GetPointData():
        point_data = unstructured_grid.GetPointData()
        reordered_point_data = vtk.vtkPointData()
        
        # Loop over each array in the point data and reorder it
        for i in range(point_data.GetNumberOfArrays()):
            array = point_data.GetArray(i)
            array_name = array.GetName()
            print(f"Reordering point data array: {array_name}")
            
            new_array = vtk.vtkFloatArray()
            new_array.SetName(array_name)
            new_array.SetNumberOfComponents(array.GetNumberOfComponents())
            new_array.SetNumberOfTuples(len(point_indices))
            
            for j, idx in enumerate(point_indices):
                new_array.SetTuple(j, array.GetTuple(idx))
            
            reordered_point_data.AddArray(new_array)
        
        structured_grid.GetPointData().ShallowCopy(reordered_point_data)

    # Write the structured grid to a new VTK file
    writer = vtk.vtkStructuredGridWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(structured_grid)
    writer.Write()

    print(f"Conversion complete. The structured VTK file is saved as '{output_file}'\n")

# Function to process all files in a given directory
def process_directory(input_dir):
    # Find all files matching the pattern "sample_*_tim_*.vtk"
    #vtk_files = glob.glob(os.path.join(input_dir, "sample_*_time_*.vtk"))
    
    vtk_files = glob.glob(os.path.join(input_dir, "sample_elasticity_2d_e*_time_*.vtk"))
    #vtk_files = glob.glob(os.path.join(input_dir, "sample_heat_2d_hc*_time_*.vtk"))
    if not vtk_files:
        print("No VTK files found matching the pattern 'sample_*_time_*.vtk' in the specified directory.")
        return

    output_dir = os.path.join(input_dir, "../vtkFiles")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(vtk_files)} files. Processing...\n")

    for input_file in vtk_files:
        # Create an output filename by prefixing "output_" and saving it in the 'converted' directory
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, f"output_{filename}")
        
        try:
            convert_unstructured_to_structured(input_file, output_file)
        except Exception as e:
            print(f"Error processing file '{input_file}': {e}\n")


def printHDF5contents(h5_file):
    with h5py.File(h5_file, 'r') as f:
        def print_attrs(name, obj):
            print(name)
            for key, val in obj.attrs.items():
                print(f"    {key}: {val}")
        f.visititems(print_attrs)




if __name__ == '__main__':
    
    #Convert unStructured vtk to structured vtk
    input_dir = "xdmf_to_vtk"
    process_directory(input_dir)

    
    # Convert VTK back to HDF5
    vtk_output_directory = 'vtkFiles'  # Replace with the path to your VTK files
    output_h5_file = 'output_elasticity_data1.h5'
    vtk_to_h5_2(vtk_output_directory, output_h5_file) #Elasticity

    # Print HDF5 file contents
    printHDF5contents(output_h5_file)
