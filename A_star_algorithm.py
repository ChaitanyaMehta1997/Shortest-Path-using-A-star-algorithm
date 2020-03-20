"""
@Author : Chaitanya Mehta
"""
import math
from PIL import Image, ImageDraw

import sys
from collections import defaultdict

from PIL import Image
import heapq


class Vertex:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.distance = float('inf')
        self.parent = None
        self.weight = 1
        self.edge = defaultdict(list)
        self.heuristic = 0


class Shortest_path:
    def __init__(self, map_image, elevations, Co_ordinates_of_start_end_points, season, output_image):
        self.season = season
        self.map_image = map_image
        self.elevations = elevations
        self.Co_ordinates_of_start_end_points = Co_ordinates_of_start_end_points
        self.output = output_image
        self.penaltyBook = {(248, 148, 18): 10, (255, 192, 0): 13, (255, 255, 255): 12, (2, 208, 60): 13,
                            (2, 136, 40): 14,
                            (5, 73, 24): float('inf'),
                            (0, 0, 255): float('inf'), (71, 51, 3): 3, (0, 0, 0): 5, (205, 0, 101): float('inf'),
                            (0,255,255):20}

    def read_input_map(self):
        x = Image.open(self.map_image).convert("RGB")
        y = x.load()

        z = x.size
        print(z)
        Input_matrix = []
        for i in range(z[0]):
            store = []
            for j in range(z[1]):
                store.append(y[i, j])

            Input_matrix.append(store)

        return Input_matrix

    def read_destination(self):
        """
        This has list containing tuples which contain the x and y points

        """
        destination_list = []
        with open(self.Co_ordinates_of_start_end_points) as fp:
            for line in fp:
                store = line.split(" ")
                destination_list.append((store[0].strip(), store[1].strip()))

        return destination_list

    def read_elevations(self):
        Elevations_matrix = []

        with open(self.elevations) as fp:
            for line in fp:
                store = []

                x = line.split()
                # Because we are suppose to skip the last 5 columns
                for a in range(0, len(x) - 5):
                    store.append(float(x[a]))

                Elevations_matrix.append(store)
        final_ev = [[Elevations_matrix[j][i] for j in range(len(Elevations_matrix))] for i in
                    range(len(Elevations_matrix[0]))]

        return final_ev

    prac_matrix = [[j for j in range(100)] for i in range(100)]

    def spring_nodes(self, matrix, input_m):
        spring_N = []
        for i in range(len(matrix) - 1):
            for j in range(len(matrix[0]) - 1):

                if input_m[i][j] == (0, 0, 255):

                    if input_m[i - 1][j] != (0, 0, 255):
                        spring_N.append(matrix[i][j])

                    if input_m[i + 1][j] != (0, 0, 255):
                        spring_N.append(matrix[i][j])

                    if input_m[i][j - 1] != (0, 0, 255):
                        spring_N.append(matrix[i][j])

                    if input_m[i][j + 1] != (0, 0, 255):
                        spring_N.append(matrix[i][j])

        return spring_N

    def mud_bfs(self, mud, matrix):
        visited = set()
        queue = []
        final_neighbours = []
        level_counter = 0
        for outer in mud:

            queue.append(outer)
            level_counter = 0
            visited.add(outer)

            while queue:
                if level_counter == 15:
                    break

                level_counter += 1
                temp = queue.pop()

                for neighbour in temp.edge.get(0):
                    if neighbour not in visited and matrix[neighbour.x][neighbour.y] != (0, 0, 255):
                        queue.append(neighbour)
                        visited.add(neighbour)
                        final_neighbours.append((temp.x, temp.y))

        return final_neighbours

    def water_nodes(self, matrix, input_m):
        water_N = []
        for i in range(len(matrix) - 1):
            for j in range(len(matrix[0]) - 1):

                if input_m[i][j] == (0, 0, 255):

                    if input_m[i - 1][j] != (0, 0, 255):
                        water_N.append(matrix[i][j])

                    if input_m[i + 1][j] != (0, 0, 255):
                        water_N.append(matrix[i][j])

                    if input_m[i][j - 1] != (0, 0, 255):
                        water_N.append(matrix[i][j])

                    if input_m[i][j + 1] != (0, 0, 255):
                        water_N.append(matrix[i][j])

        return water_N

    def water_ice_bfs(self, water, matrix, input_matrix):
        visited = set()
        queue = []
        final_neighbours = []
        level_counter = 0
        for outer in water:

            queue.append(outer)
            level_counter = 0
            visited.add((outer.x, outer.y))

            while queue:
                if level_counter == 1:
                    break

                level_counter += 1
                temp = queue.pop()

                for neighbour in temp.edge.get(0):
                    if (neighbour.x, neighbour.y) not in visited and input_matrix[neighbour.x][neighbour.y] != (0, 0, 255):

                        queue.append(neighbour)
                        visited.add((neighbour.x, neighbour.y))
                        final_neighbours.append((neighbour.x, neighbour.y))

        return final_neighbours

    def Create_graph(self, matrix, s, d):
        visited = set()
        book = {}
        object_store = []
        priority_q = []
        s = ""
        Elevations = self.read_elevations()
        elevation_checker = 0
        is_elevation_pos = False
        is_elevation_neg = False
        elevaion_constant_positive = 2
        elevaion_constant_negative = 0.5
        c_c = 0
        for i in range(0, len(matrix)):
            store2 = []
            for j in range(0, len(matrix[0])):

                is_elevation_pos = False
                is_elevation_neg = False

                if Elevations[i][j] > elevation_checker:
                    elevation_checker = Elevations[i][j]
                    is_elevation_pos = True

                if Elevations[i][j] < elevation_checker:
                    elevation_checker = Elevations[i][j]
                    is_elevation_neg = True

                if s == "Fall" or s == "fall":

                    if self.penaltyBook.get(matrix[i][j]) == 7:
                        if is_elevation_pos:

                            c_c = (self.penaltyBook.get(matrix[i][j]) + 10) * elevaion_constant_positive

                        elif is_elevation_neg:
                            c_c = (self.penaltyBook.get(matrix[i][j]) + 10) * elevaion_constant_negative

                        else:
                            c_c = (self.penaltyBook.get(matrix[i][j]))


                else:
                    if is_elevation_pos:

                        c_c = (self.penaltyBook.get(matrix[i][j])) * elevaion_constant_positive

                    elif is_elevation_neg:
                        c_c = (self.penaltyBook.get(matrix[i][j])) * elevaion_constant_negative

                    else:
                        c_c = (self.penaltyBook.get(matrix[i][j]))

                eucli = math.sqrt((i - d[0]) ** 2 + (j - d[1]) ** 2)

                Heuristic = eucli + c_c

                temp = Vertex(i, j)
                temp.heuristic = Heuristic
                store2.append(temp)

            object_store.append(store2)

        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[0])):

                # check neighbours

                if i - 1 >= 0:
                    object_store[i][j].edge[0].append(object_store[i - 1][j])

                if i + 1 <= len(matrix) - 1:
                    object_store[i][j].edge[0].append(object_store[i + 1][j])

                if j - 1 >= 0:
                    object_store[i][j].edge[0].append(object_store[i][j - 1])

                if j + 1 <= len(matrix[0]) - 1:
                    object_store[i][j].edge[0].append(object_store[i][j + 1])

                if i + 1 <= len(matrix) - 1 and j + 1 <= len(matrix[0]) - 1:
                    object_store[i][j].edge[0].append(object_store[i + 1][j + 1])

                if i - 1 >= 0 and j - 1 >= 0:
                    object_store[i][j].edge[0].append(object_store[i - 1][j - 1])

                if i - 1 >= 0 and j + 1 <= len(matrix[0]) - 1:
                    object_store[i][j].edge[0].append(object_store[i - 1][j + 1])

                if i + 1 <= len(matrix) - 1 and j - 1 >= 0:
                    object_store[i][j].edge[0].append(object_store[i + 1][j - 1])

        return object_store

    def A_star(self, source, destination, season):
        """
        Iterate over the the brown.txt and pick out source and destinations for A*
        :param source:
        :param destination:
        :return:
        """
        im = Image.open("terrain.png")

        input_matrix = self.read_input_map()

        k = list(list(input_matrix))

        graph = self.Create_graph(input_matrix, season, destination)

        water_vert = self.water_nodes(graph, input_matrix)

        color_ice = self.water_ice_bfs(water_vert, graph,input_matrix)

        priority_queue = []
        visited = set()

        for x in range(len(graph[0])):
            for y in range(len(graph)):

                if x == source[0] and y == source[1]:
                    graph[x][y].distance = 0
                    priority_queue.append((x, y))
                    heapq.heapify(priority_queue)
                    break

        final = []
        found = False
        while len(priority_queue) > 0:

            # if (len(visited) == len(graph)):
            #    break

            temp = heapq.heappop(priority_queue)
            smallest = graph[temp[0]][temp[1]]
            if temp[0] == destination[0] and temp[1] == destination[1]:
                print("FOUND")
                found = True

                break
            visited.add((smallest.x, smallest.y))
            final.append((smallest.x, smallest.y))

            if graph[smallest.x][smallest.y].edge.get(0):
                for a_star in graph[smallest.x][smallest.y].edge.get(0):
                    if (a_star.x, a_star.y) not in visited:
                        candidate = smallest.distance + a_star.weight + a_star.heuristic

                        if candidate < a_star.distance:
                            a_star.distance = candidate
                            a_star.parent = smallest
                            heapq.heappush(priority_queue, (a_star.x, a_star.y))

        i = destination[0]
        j = destination[1]
        z = graph[i][j]
        final_store = []
        while z.parent and found:
            print(z.x, z.y)

            a = int(z.x)
            b = int(z.y)
            final_store.append((a, b))
            k[a][b] = (238, 130, 238)

            z = z.parent
        final_store.append((source[0], source[1]))

        return final_store, found, color_ice


def main():
    '''
    terrain_image = sys.argv[1]

    elevation_file = sys.argv[2]

    path_file = sys.argv[3]

    season = sys.argv[4]

    output_image_filename = sys.argv[5]
    '''

    terrain_image = "terrain.png"

    elevation_file = "elevations.txt"

    path_file = "brown.txt"

    season = "Winter"

    output_image_filename = "xyz.png"

    final = None

    im = Image.open(terrain_image)

    draw = ImageDraw.Draw(im)

    path = Shortest_path(terrain_image, elevation_file, path_file, season, output_image_filename)

    destination_array = path.read_destination()

    for node in range(len(destination_array) - 1):
        print("This is the shortest path found by A*-------------------")
        source = [int(destination_array[node][0]), int(destination_array[node][1])]

        destination = [int(destination_array[node + 1][0]), int(destination_array[node + 1][1])]

        x = Image.open("terrain.png").convert("RGB")
        (final, found, ice) = path.A_star(source, destination, season)

        final.reverse()

        draw.line(final, fill="red", width=2)
        draw.point(ice, fill=(0,255,255))

    im.save(output_image_filename, "PNG")

    # plt.imshow(image1)
    # plt.show()


if __name__ == "__main__":
    main()
