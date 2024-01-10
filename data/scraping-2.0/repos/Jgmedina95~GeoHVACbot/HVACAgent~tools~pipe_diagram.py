from langchain.tools import BaseTool
import matplotlib.pyplot as plt
import numpy as np

class make_diagram_fxns:
        
    def query_to_dict(self,query_str):
        # Split from commas
        query_list = query_str.split(',')
        # Split from equal sign and strip spaces
        query_dict = {q.split('=')[0].strip(): q.split('=')[1].strip() for q in query_list}
        return query_dict

 
    def draw_utubes(self,num_tubes,depth):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        def draw_utube(ax, start_y,depth):
            x = [0, 0, 0, 0]
            y = [start_y, start_y, start_y, start_y]
            z = [0, -depth, -depth, 0]
            ax.plot(x, y, z, linewidth=3)

        for i in range(num_tubes):
            draw_utube(ax, i*2,depth/10)

        x = np.linspace(-1, 5, 2)
        y = np.linspace(0, 1 + 2*num_tubes, num_tubes + 1)  # Adjusted to start from y=0
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)
        ax.plot_surface(X, Y, Z, alpha=0.5, color='gray')

        # Adjusted the y_wall values to ensure it starts from the floor
        y_wall = np.linspace(-1, 1 + 2*num_tubes, num_tubes + 1)
        z_wall = np.linspace(0, 3, 2)
        Y_wall, Z_wall = np.meshgrid(y_wall, z_wall)
        X_wall = np.ones(Y_wall.shape) * -1
        ax.plot_surface(X_wall, Y_wall, Z_wall, alpha=0.5, color='gray')

        # Adjusted window boundaries to ensure they lie within the wall
        y_window = np.linspace(4, 6, 2)
        z_window = np.linspace(1, 2, 2)
        Y_window, Z_window = np.meshgrid(y_window, z_window)
        X_window = np.ones(Y_window.shape) * -1
        ax.plot_surface(X_window, Y_window, Z_window, color='blue')

        max_range = np.array([1.0, 1 + 2*num_tubes, 3.0]).ptp()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten()
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten()
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten()

        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')


        ax.set_zlabel('fts *10',rotation=90)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Diagram for {num_tubes} pipes vertical design")
        plt.show()


class make_diagram(BaseTool):
    name = "make diagrams"
    description = """This tool will make diagrams for your project. Based on the pipe system arrangement.
    input: (Keyword arguments) length in feets, num_tubes"""


    def _run(self, query):
        fxns = make_diagram_fxns()
        query_params = fxns.query_to_dict(query)
        num_tubes = query_params.get('num_tubes', -1)
        depth = query_params.get('depth', -1)
        if num_tubes == -1:
            return "Please enter params correctly. Example: num_tubes=4, depth=100"
        elif depth == -1:
            return "Please enter params correctly. Example: num_tubes=4, depth=100"
        else:
            return fxns.draw_utubes(int(num_tubes),int(depth))

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")