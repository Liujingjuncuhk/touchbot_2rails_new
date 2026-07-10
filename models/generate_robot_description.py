import numpy as np
import pickle
import pyvista as pv

def generate_robot_description(pickleFilename: str, robot_model: str, pullpoints, pulley_location):
    """
    Generates a robot description file in the specified format.

    Parameters:
    - filename: The name of the file to save the robot description.
    - robot_model: The model of the robot.
    - robot_color: The color of the robot in hexadecimal format (e.g., '#FF5733').
    """
    with open(robot_model, 'r') as f:
        num_vert, num_tet = map(int, f.readline().split())
        vertices = np.array([list(map(float, f.readline().split())) for _ in range(num_vert)])
        tetrahedra = np.array([list(map(int, f.readline().split())) for _ in range(num_tet)])
    vertices = vertices * 1e-3
    nCable = len(pullpoints)
    pp_idx = [0 for _ in range(len(pullpoints))]
    minx = np.min(vertices[:,0])
    maxx = np.max(vertices[:,0])
    miny = np.min(vertices[:,1])
    maxy = np.max(vertices[:,1])
    for i, pp in enumerate(pullpoints):
        # find the closest vertex index that is not on edges
        distances = np.linalg.norm(vertices - pp, axis=1)
        mask = (vertices[:,0] > minx+1e-4) & (vertices[:,0] < maxx-1e-4) & (vertices[:,1] > miny+1e-4) & (vertices[:,1] < maxy-1e-4)
        distances[~mask] = np.inf
        pp_idx[i] = np.argmin(distances)
        vertices[pp_idx[i]] = pp  # set the pullpoint vertex to exact location
    print(f"Pullpoints indices: {pp_idx}")

    # rotate the vertices along z-axis for 15 degree
    # theta = np.deg2rad(-15)
    # rot_z = np.array([[np.cos(theta), -np.sin(theta), 0],
    #                   [np.sin(theta), np.cos(theta), 0],
    #                   [0, 0, 1]])

    # vertices = vertices @ rot_z.T
    print("distance on z axis is: ")
    print(np.max(vertices[:,2]) - np.min(vertices[:,2]))

    mesh = pv.PolyData(vertices)
    mesh.faces = np.hstack([[4, *tet] for tet in tetrahedra])
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='lightblue', show_edges=True)
    
    # Add pullpoints
    pullpoints = vertices[pp_idx]
    plotter.add_points(pullpoints, color='red', point_size=10, render_points_as_spheres=True)
    
    # Add pulley locations
    plotter.add_points(pulley_location, color='green', point_size=10, render_points_as_spheres=True)

    # Add lines
    for i in range(nCable):
        plotter.add_lines(np.array([pulley_location[i], vertices[pp_idx[i]]]), color='green')
    plotter.show_grid()
    plotter.show_axes()
    # make axis equal
    plotter.set_scale(1, 1, 1)
    plotter.show()

    # write each input to a pickle file in the folderName
    with open(pickleFilename, 'wb') as f:
        pickle.dump({
            'vertices': vertices,
            'tetrahedra': tetrahedra,
            'pp_idx': pp_idx,
            'pulley_location': pulley_location
        }, f)
    

if __name__ == "__main__":
    pickleFilename = 'palm_size3.pickle'
    robot_model = 'models/palm_size3.tet'
    pulley_location = np.array([[60.93, -117.64, 2.5], [100.93, -117.64, 2.5],[100.93, -117.64, 62.5], [60.93, -117.64, 62.5]]) *1e-3
    # pullpoints = np.array([[61,-96, 0],[61, 96,  0], [-61, -96,  0], [ -61,96, 0], [61, -26, 3],[61,26, 3],[-61,-26, 3],[-61,26, 3]])*1e-3
    # pulley_locations = np.array([[444,724,46],[444,36,46],[116,724,46],[116,36,46],[464,724,561],[464,36,561],[96, 724,561],[96, 36,561]])*1e-3
    pplocation_size3 = np.array([[57.61, 8.96, 6], [65.12, 46.49, 8.00], [65.12, 46.49, 57], [57.61, 8.96, 60]])*1e-3
    # pplocation_size2 = np.array([[57.76, 9.89, 4.5], [52.66, 47.39, 8], [52.66, 47.39, 57], [57.61,  9.89, 60.5]])*1e-3
    
    generate_robot_description(pickleFilename, robot_model, pplocation_size3, pulley_location)
    print(f"Robot description saved to {pickleFilename}")


    
