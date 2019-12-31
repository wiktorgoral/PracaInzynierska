import numpy as np
import matplotlib.pyplot as plt


class Node:

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def return_path(current_node, maze):
    path = []
    no_rows, no_columns = np.shape(maze) 
    # Tablica wyjsciowa z wartosciami - 1
    result = [[-1 for i in range(no_columns)] for j in range(no_rows)]
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent

    path = path[::-1]
    start_value = 0 
    # Ustawienie wartosci węzłów scieżki
    for i in range(len(path)):
        result[path[i][0]][path[i][1]] = start_value
        start_value += 1
    return result


def search(maze, cost, start, end):
    start_node = Node(None, tuple(start))
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, tuple(end))
    end_node.g = end_node.h = end_node.f = 0

    # Tablica nie odkrytych wezłów
    yet_to_visit_list = [] 
    # Tablica odkrytych węzłów
    visited_list = []

    # Dodanie węzła startowego do listy do odwiedzenia
    yet_to_visit_list.append(start_node)

    # Zabezpieczenie w razie nieskończonej pętli
    outer_iterations = 0
    max_iterations = (len(maze) // 2) ** 10

    # Możliwe ruchy
    move = [
        [-1, 0],
        [0, -1],
        [1, 0],
        [0, 1]
    ]
    no_rows, no_columns = np.shape(maze)
    
    # Główna pętla
    while len(yet_to_visit_list) > 0:

        outer_iterations += 1
        current_node = yet_to_visit_list[0]
        current_index = 0
        for index, item in enumerate(yet_to_visit_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

            # warunek zabezpieczajacy
            if outer_iterations > max_iterations:
                print("Za dużo iteracjii")
                return return_path(current_node, maze)

        # Zmiana węzła na odwiedzone
        yet_to_visit_list.pop(current_index)
        visited_list.append(current_node)

        # Sprawdzenie czy koniec został osiągnięty
        if current_node == end_node:
            return return_path(current_node, maze)

        # Tablica dziedziczących węzłów
        children = []

        for new_position in move:

            # Położenie aktualnego węzłów
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Warunek na maxymalne położenie
            if (node_position[0] > (no_rows - 1) or node_position[0] < 0 or node_position[1] > (no_columns - 1) or
                    node_position[1] < 0):
                continue

            # Warunek na możliwosć przejscia
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            new_node = Node(current_node, node_position)

            # Dodanie do tablicy pobliskich węzełów
            children.append(new_node)

        # Pętla do sprawdzenia dziedziczących węzłów
        for child in children:

            # Dziedziczący jest na liscie odwiedzonych węzłów
            if len([visited_child for visited_child in visited_list if visited_child == child]) > 0:
                continue

            child.g = current_node.g + cost 
            # Wyliczenie kosztu przejscia przy użyciu euklidesowej odległości
            child.h = (((child.position[0] - end_node.position[0]) ** 2) +
                       ((child.position[1] - end_node.position[1]) ** 2))

            child.f = child.g + child.h

            # Węzeł jest już w liscie do odwiedzenia i ma mniejszy koszt
            if len([i for i in yet_to_visit_list if child == i and child.g > i.g]) > 0:
                continue
            yet_to_visit_list.append(child)


if __name__ == '__main__':
    maze = [
        [0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0]
    ]
    # Węzeł startowa
    start = [0, 0] 
    # Węzeł końcowa
    end = [4, 5]  
    # Koszt ruchu
    cost = 1   

    path = search(maze, cost, start, end)

    # Wyswietlenie efektów
    plt.imshow(maze)
    plt.show()
    plt.imshow(path)
    plt.show()