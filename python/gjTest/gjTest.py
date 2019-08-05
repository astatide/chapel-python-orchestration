#import sys
#import importlib
#import faulthandler
#faulthandler.enable()
import subprocess
import gc
gc.disable()
import gjallarbru as gj
import os
import random
import numpy as np
import time
import copy
# Shut tensorflow up.
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
# for now, kill stdout
import sys
#stdout = sys.stdout
#sys.stdout = open('/dev/null', 'w')
#stderr = sys.stderr
#sys.stderr = open(os.devnull, 'w')
#import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.models import Model
from keras.initializers import VarianceScaling, Zeros, RandomNormal

class Tartarus():
    def create_empty_board(self, rows, columns):
        ''' Creates empty Tartarus board '''
        board = []

        for row in range(rows):
            board_row = []
            for column in range(columns):
                board_row.append(self.SPACE_EMPTY)
            board.append(board_row)

        return board


    def initialize_placements_on_board(self, is_box):
        is_legal_position = False

        while (is_legal_position == False):
            row = random.randint(1, self.ROWS - 2) #boxes are only placed in inner positions for tartarus problem
            col = random.randint(1, self.COLUMNS - 2)

            if(self.is_position_occupied(row,col) == False):
                if(is_box == True):
                    if(self.would_box_placement_make_invalid_start(row,col) == False):
                        self.board[row][col] = self.SPACE_BOX
                        is_legal_position = True
                else:
                    self.board[row][col] = self.SPACE_DOZER
                    self.cur_dozer_location = (row, col)
                    rand_move_dir = random.randint(0, 3)
                    self.cur_dozer_forward_index = rand_move_dir  #random pick of which move forward direction
                    is_legal_position = True

    def is_position_occupied(self,row, col):
        if(self.board[row][col] == self.SPACE_EMPTY):
            return False
        else:
            return True

    def would_box_placement_make_invalid_start(self,row, col):
        '''Approx. 7.3% of all possible starting configurations exhibit this, need to avoid:
        Unsolvable starting configurations exist: if four or more adjacent boxes form a
        rectangular block, it becomes impossible for the robot to push any of these boxes
        and the maximum achievable score drops

        OX  XO  XX  XX
        XX  XX  OX  XO

        Would be invalid to place in "O"

        '''

        #iterate directions clockwise from potential placement "O", if 3 in a row are boxes "X" then invalid
        box_count = 0
        clockwise_rotation = [0,1,2,4,7,6,5,3,0,1]
        rotation_idx = 0
        while rotation_idx < len(clockwise_rotation):
            cw_idx = clockwise_rotation[rotation_idx]
            dr, dc = self.DIRECTIONS[cw_idx]
            r = row + dr
            c = col + dc

            if(self.board[r][c] == self.SPACE_BOX):
                box_count += 1
                if(box_count == 3):
                    return True
            else:
                box_count = 0

            rotation_idx += 1

        return False


    def print_board(self):
        ''' Prints Tartarus board '''
        for row in self.board:
            print ('|' + '|'.join(row) + '|')


    def rotate_dozer_direction(self, rotation):
        if(self.cur_dozer_forward_index == 0): #Pointing North
            if(rotation == "Left"):
                self.cur_dozer_forward_index = 1
            else:
                self.cur_dozer_forward_index = 2
        elif (self.cur_dozer_forward_index == 1): #Pointing West
            if (rotation == "Left"):
                self.cur_dozer_forward_index = 3
            else:
                self.cur_dozer_forward_index = 0
        elif (self.cur_dozer_forward_index == 2): #Pointing East
            if (rotation == "Left"):
                self.cur_dozer_forward_index = 0
            else:
                self.cur_dozer_forward_index = 3
        else:                                        #Pointing South
            if (rotation == "Left"):
                self.cur_dozer_forward_index = 2
            else:
                self.cur_dozer_forward_index = 1

    def check_unique_space(self, row, col):
        key = str(row) + str(col)
        if key in self.unique_register:
            return False
        else:
            self.unique_register[key] = 0
            self.unique_box_visit_count += 1
            return True
    def get_unique_box_score(self):
        return self.unique_box_visit_count

    def try_move_forward(self):
        dr, dc = self.DOZER_TURN_MOVE_DIRECTIONS[self.cur_dozer_forward_index]
        row, col = self.cur_dozer_location
        r = row + dr
        c = col + dc


        if (r < 0 or c < 0 or r > self.ROWS - 1 or c > self.COLUMNS - 1):
            one_move_status = self.SPACE_WALL
        else:
            one_move_status = self.board[r][c]

        if (one_move_status == self.SPACE_EMPTY):
            self.board[row][col] = self.SPACE_EMPTY
            self.board[r][c] = self.SPACE_DOZER
            self.cur_dozer_location = (r, c)
            self.check_unique_space(r, c)
            return True
        else:
            r2 = row + dr * 2
            c2 = col + dc * 2

            if (r2 < 0 or c2 < 0 or r2 > self.ROWS - 1 or c2 > self.COLUMNS - 1):
                two_move_status = self.SPACE_WALL
            else:
                two_move_status = self.board[r2][c2]

            if (two_move_status == self.SPACE_EMPTY):
                if (one_move_status == self.SPACE_BOX):
                    self.board[row][col] = self.SPACE_EMPTY
                    self.board[r][c] = self.SPACE_DOZER
                    self.cur_dozer_location = (r, c)
                    self.check_unique_space(r, c)

                    self.board[r2][c2] = self.SPACE_BOX

            elif (two_move_status == self.SPACE_BOX):
                return False
            elif (two_move_status == self.SPACE_WALL):
                return False

    def get_board(self):
        return self.board

    def set_board(self, board):
        self.board = board[0]
        self.cur_dozer_location = board[1]
        self.cur_dozer_forward_index = board[2]
        self.unique_register = {}
        self.unique_box_visit_count = 0

    def init_new_borad(self):
        self.board = self.create_empty_board(self.ROWS, self.COLUMNS)
        for box in range(self.BOX_COUNT):
            self.initialize_placements_on_board(True)  # Boxes
        self.initialize_placements_on_board(False)  # Dozer
        self.unique_register = {}
        self.unique_box_visit_count = 0

        return (self.board, self.cur_dozer_location, self.cur_dozer_forward_index)

    def get_state_from_sensors(self):

        #clockwise_rotation = [0, 1, 2, 4, 7, 6, 5, 3]

        if (self.cur_dozer_forward_index == 0):  # Pointing North
            clockwise_rotation = [0, 1, 2, 4, 7, 6, 5, 3]
        elif (self.cur_dozer_forward_index == 1):  # Pointing West
            clockwise_rotation = [3, 0, 1, 2, 4, 7, 6, 5]
        elif (self.cur_dozer_forward_index == 2):  # Pointing East
            clockwise_rotation = [4, 7, 6, 5, 3, 0, 1, 2]
        else:                                       # Pointing South
            clockwise_rotation = [7, 6, 5, 3, 0, 1, 2, 4]

        row, col = self.cur_dozer_location
        rotation_idx = 0
        state = []

        while rotation_idx < len(clockwise_rotation):
            cw_idx = clockwise_rotation[rotation_idx]
            dr, dc = self.DIRECTIONS[cw_idx]
            r = row + dr
            c = col + dc

            if (r < 0 or c < 0 or r > self.ROWS - 1 or c > self.COLUMNS - 1):
                #wall\
                #state.append(self.SPACE_WALL)
                state.append(0)

            else:
                location_state = self.board[r][c]
                if (location_state == self.SPACE_EMPTY):
                    #state.append(self.SPACE_EMPTY)
                    state.append(1)
                else:
                    #state.append(self.SPACE_BOX)
                    state.append(2)

            rotation_idx += 1

        return state

    def get_final_val(self, corner_val, edge_val):
        final_val = 0
        self.CORNER_VAL = corner_val
        self.EDGE_VAL = edge_val

        #check corners
        if (self.board[0][0] == self.SPACE_BOX):
            final_val += self.CORNER_VAL
        if(self.board[0][self.COLUMNS - 1] == self.SPACE_BOX):
            final_val += self.CORNER_VAL
        if (self.board[self.ROWS - 1][0] == self.SPACE_BOX):
            final_val += self.CORNER_VAL
        if (self.board[self.ROWS - 1][self.COLUMNS - 1] == self.SPACE_BOX):
            final_val += self.CORNER_VAL


        #check edges minus corners
        for i in range(1, self.COLUMNS - 1):
            if (self.board[0][i] == self.SPACE_BOX):
                final_val += self.EDGE_VAL
            if (self.board[self.ROWS - 1][i] == self.SPACE_BOX):
                final_val += self.EDGE_VAL

        for i in range(1, self.ROWS - 1):
            if (self.board[i][0] == self.SPACE_BOX):
                final_val += self.EDGE_VAL
            if (self.board[i][self.COLUMNS - 1] == self.SPACE_BOX):
                final_val += self.EDGE_VAL

        return final_val

    def get_final_board_state_as_vector(self):
        state_vec = []

        for row in self.board:
            for col in row:
                if col == self.SPACE_BOX:
                    state_vec.append(1)
                else:
                    state_vec.append(0)

        return state_vec

    def get_inner_box_box_score(self, inner_box_desired_count):
        #set inner_box_desired_count to 2 check for a behavior of only 2 boxes in the inner box - behavior for deceptive tartarus
        inner_box_box_score = 6
        inner_box_box_count = 0
        corner_bonus = 0

        for i in range(1, self.ROWS - 1):
            for j in range(1, self.COLUMNS - 1):
                if (self.board[i][j] == self.SPACE_BOX):
                    inner_box_box_count += 1



        if inner_box_box_count < inner_box_desired_count:
            inner_box_box_score = 0 + inner_box_box_count
        elif inner_box_box_count > inner_box_desired_count:
            inner_box_box_score = inner_box_box_score - inner_box_box_count


        if inner_box_box_score == 6:
            # check corners
            if (self.board[0][0] == self.SPACE_BOX):
                corner_bonus += 1
            if (self.board[0][self.COLUMNS - 1] == self.SPACE_BOX):
                corner_bonus += 1
            if (self.board[self.ROWS - 1][0] == self.SPACE_BOX):
                corner_bonus += 1
            if (self.board[self.ROWS - 1][self.COLUMNS - 1] == self.SPACE_BOX):
                corner_bonus += 1

        inner_box_box_score += corner_bonus

        return inner_box_box_score

    def get_final_distance_from_center_score(self):
        final_dist_val = 0
        row_idx = 0
        while row_idx < self.ROWS:
            col_idx = 0
            while col_idx < self.COLUMNS:
                if (self.board[row_idx][col_idx] == self.SPACE_BOX):
                    final_dist_val = final_dist_val + self.dispersal_distance_bc_score[str(row_idx) + str(col_idx)]
                col_idx += 1
            row_idx += 1

        return final_dist_val

    def create_dispersal_dist_score_matrix(self):
        #dynamically created based on number of cols and rows
        row_idx = 0
        dispersal_bc_score = {}
        row_half_count = 0
        half_val = (self.ROWS/2)
        prev_start_val = 0

        while row_idx < self.ROWS:
            half_count = 0
            col_idx = 0
            if row_half_count < half_val:
                start_val = ((self.ROWS / 2) + 1) - row_idx
                prev_start_val = start_val
                row_half_count += 1
            elif row_half_count == half_val:
                row_half_count += 1
                start_val = prev_start_val
                prev_start_val += 1
            else:
                start_val = prev_start_val
                prev_start_val += 1

            while col_idx < self.COLUMNS:
                dispersal_bc_score[str(row_idx) + str(col_idx)] = start_val
                if half_count <  half_val - 1:
                    start_val = start_val - 1
                    half_count += 1
                elif half_count ==  half_val - 1:
                    half_count += 1
                else:
                    start_val = start_val + 1
                col_idx += 1
            row_idx += 1
        return dispersal_bc_score

    def __init__(self):
        self.ROWS = 6
        self.COLUMNS = 6
        self.BOX_COUNT = 6

        self.CORNER_VAL = 2
        self.EDGE_VAL = 0

        self.SPACE_EMPTY = ' '
        self.SPACE_BOX = 'B'
        self.SPACE_DOZER = 'D'
        self.SPACE_WALL = "W"

        self.dispersal_distance_bc_score  = self.create_dispersal_dist_score_matrix()

        self.DIRECTIONS = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        )
        self.DOZER_TURN_MOVE_DIRECTIONS = (
                    (-1, 0),
            (0, -1),           (0, 1),
                    (1, 0),
        )
        self.cur_dozer_forward_index = 0
        self.cur_dozer_location = (0,0)

        self.unique_register = {}
        self.unique_box_visit_count = 0


        self.board = self.create_empty_board(self.ROWS,self.COLUMNS)

        random.seed(int(time.time()))
        for box in range(self.BOX_COUNT):
            self.initialize_placements_on_board(True) #Boxes
        self.initialize_placements_on_board(False) #Dozer

class Evaluate:
    def nomalize_deme_val(self, deme_idx, deme_val):
        min = 0
        max = 0

        if deme_idx == 0:
            #cumulative_val = cumulative_val + tartarus.get_final_val(2, -1)
            min = -6
            max = 8
        elif deme_idx == 1:
            # cumulative_val = cumulative_val + tartarus.get_final_val(2, 1)
            min = 0
            max = 10
        elif deme_idx == 2:
            #cumulative_val = cumulative_val + tartarus.get_final_distance_from_center_score()
            min = 3
            max = 22
        elif deme_idx == 3:
            #cumulative_val = cumulative_val + tartarus.get_unique_box_score()
            min = 0
            max = 36
        elif deme_idx == 4:
            #cumulative_val = cumulative_val + tartarus.get_inner_box_box_score()
            min = -8
            max = 2

        z = (deme_val - min) / (max - min)

        return z

    def evaluate_model(self, indv_model_expressed, model_eval_pop, GEP_HEAD_LEN, model_input_val, model_output_val):
        top_fitness_val = 0.0
        top_fitness_idx = 1
        top_fitness_indices = []
        idx = 0


        print("model: " + str(indv_model_expressed))
        for gene in indv_model_expressed:
            normalized_results = 0
            for indv in model_eval_pop:
                deme_idx = indv[2]
                indv_fitness_val, indv_fitness_indices, final_state = self.evaluate_chromosome(indv[0], indv[1], GEP_HEAD_LEN, deme_idx, gene, model_input_val, model_output_val)

                norm_val = self.nomalize_deme_val(deme_idx, indv_fitness_val)
                normalized_results += norm_val
            normalized_fitness = normalized_results / len(model_eval_pop) #find the mean of the normalized results to compare across evaluations

            if normalized_fitness > top_fitness_val: # for tied results if you do >= it takes later more complext models, > is first model
                top_fitness_val = copy.deepcopy(normalized_fitness)
                top_fitness_idx = copy.deepcopy(idx)
                #print(str(top_fitness_idx))

            idx += 1

        print("model result: " + str(top_fitness_val) + " index: " + str(top_fitness_indices.append(top_fitness_idx)))

        return top_fitness_val, top_fitness_indices.append(top_fitness_idx)

    def chromosome_embeded_indices(self, expressed, chrm_indices):
        embedded = False

        for idx, gene in enumerate(expressed):
            if gene[0] == 0:
                chrm_indices.append(idx)
                embedded = True

        return embedded

    def get_embeded_chromosomes(self,expressed, chrm_indices, comb_truncation, chrm_expressed_order):
        embeded_chromosomes = []

        for idx in chrm_indices:
            id = expressed[idx][1]
            chrm_expressed_order.append(id)
            chrm = comb_truncation[id]
            embeded_chromosomes.append(chrm)

        return embeded_chromosomes

    def express_embeded_chromosomes(self, embeded_chromosomes, GEP_HEAD_LEN, expressions, combinatorials):

        for chrm in embeded_chromosomes:
            express = express_chrm.Expression()
            indv_expressed, indv_combinatorial_idxs = express.express_chromosome(chrm, GEP_HEAD_LEN)
            expressions[chrm.id] = indv_expressed
            combinatorials[chrm.id] = indv_combinatorial_idxs

        return expressions, combinatorials

    def load_embedded_chromosome(self, expressed, combinatorial_idxs, model_layer_units, model_input_val, model_output_val, tmodel_class, best_index, chrm_id):
        for idx, gene in enumerate(expressed):

            if idx in combinatorial_idxs:
                save_weights = True
            else:
                save_weights = False

            calc_new_weights = isinstance(gene[0], int)  # vs using saved weights
            model = self.load_model(gene, save_weights, idx, model_layer_units, model_input_val, model_output_val,
                                    calc_new_weights, tmodel_class)

            if idx == best_index:
                tmodel_class.load_weights(model.get_weights(), chrm_id)


    def load_model(self, gene, save_weights, idx, model_layer_units, model_input_val, model_output_val, calc_new_weights, tmodel_class, load_partial=False):


        if calc_new_weights == True:
            if gene[0] == 0:
                model = tmodel_class.build_model([123], True, idx, model_layer_units, model_input_val,
                                                 model_output_val, self.GAME, gene[1])  # pass dummy init seed
            else:
                # gene is list of seeds
                model = tmodel_class.build_model(gene, save_weights, idx, model_layer_units, model_input_val,
                                                 model_output_val, self.GAME)
        else:
            # gene is combinatoral function
            model = tmodel_class.build_model([123], save_weights, idx, model_layer_units, model_input_val,
                                             model_output_val, self.GAME, gene)  # pass dummy init seed

        return model

    def save_model_weights(self, weights, id, tmodel_class):
        tmodel_class.load_weights(weights, id)

    def clear_model(self, tmodel_class):
        tmodel_class.clear_model_session()

    def evaluate_chromosome(self, expressed, combinatorial_idxs, GEP_HEAD_LEN, deme_idx, model_layer_units, model_input_val, model_output_val, comb_truncation):

        tmodel_class = Tar_Model.Tartarus_Model(self.WEIGHT_NOISE_MU, self.WEIGHT_NOISE_SIGNMA)

        top_fitness_val = 0
        top_val_len = math.inf
        top_fitness_indices = []
        top_val_final_state = []
        best_index = 0
        expressions = {}
        combinatorials = {}
        chrm_indices = []
        chrm_expressed_order = []

        #print(expressed)

        self.chromosome_embeded_indices(expressed, chrm_indices)
        while chrm_indices:
            embeded_chromosomes = self.get_embeded_chromosomes(expressed, chrm_indices, comb_truncation, chrm_expressed_order)
            self.express_embeded_chromosomes(embeded_chromosomes, GEP_HEAD_LEN, expressions, combinatorials)
            # for idx in chrm_indices:
            #     chrm_expressed_order.append(idx)

            chrm_indices = []
            for key, expression in expressions.items():
                self.chromosome_embeded_indices(expression, chrm_indices)

        #now load items in reverse order
        i = len(chrm_expressed_order) - 1
        while i >= 0:
            chrm_id = chrm_expressed_order[i]
            best_idx = comb_truncation[chrm_id].best_index
            self.load_embedded_chromosome(expressions[chrm_id], combinatorials[chrm_id], model_layer_units, model_input_val,
                                     model_output_val, tmodel_class, best_idx, chrm_id)



            i -= 1

        idx = 0
        for gene in expressed:
            #print(gene)
            if idx in combinatorial_idxs:
                save_weights = True
            else:
                save_weights = False

            start_time = time.time()

            # is_seed_val_int = isinstance(cur_seed, int)
            # if is_seed_val_int == True:

            calc_new_weights = isinstance(gene[0], int) #vs using saved weights
            model = self.load_model(gene, save_weights, idx, model_layer_units, model_input_val, model_output_val, calc_new_weights, tmodel_class)

            # if is_GEP == True:
            #     #gene is list of seeds
            #     model = tmodel_class.build_model(gene, save_weights, idx, model_layer_units, model_input_val, model_output_val, self.GAME)
            # else:
            #     #gene is combinatoral function
            #     model = tmodel_class.build_model([123], save_weights, idx, model_layer_units, model_input_val, model_output_val, self.GAME,  gene) #pass dummy init seed

            # print("--- Model %s---" % (time.time() - start_time))
            # start_time = time.time()

            final_val, final_state = self.run_eval_on_model(model, deme_idx)
            #print(final_val)

            # print("--- Run Eveal %s---" % (time.time() - start_time))
            # start_time = time.time()

            #check to see len of gene, shorter one win in a tied score
            #if combinatorial, then store both?

            replace = False

            if final_val > 0:
                if calc_new_weights == True:
                    if final_val > top_fitness_val:
                        replace = True
                    elif final_val == top_fitness_val:
                        if len(gene) < top_val_len:
                            replace = True

                    if replace == True:
                        top_val_len = len(gene)
                        top_fitness_val = copy.deepcopy(final_val)
                        top_val_final_state = copy.deepcopy(final_state)
                        top_fitness_indices = []
                        top_fitness_indices.append(idx)
                        best_index = idx
                else:
                    if final_val > top_fitness_val:
                        best_index = idx
                        g1 = gene[1]
                        g2 = gene[2]

                        linked_all = False

                        top_fitness_val = copy.deepcopy(final_val)
                        top_val_final_state = copy.deepcopy(final_state)
                        top_fitness_indices = []
                        top_fitness_indices.append(g1)
                        top_fitness_indices.append(g2)


                        for index in top_fitness_indices:
                            if index > (GEP_HEAD_LEN + 1): #another MEP
                                gene_exp = expressed[index]
                                top_fitness_indices.append(gene_exp[1])
                                top_fitness_indices.append(gene_exp[2])


            #clear session before next gene sequence
            self.clear_model(tmodel_class)
            #tmodel_class.clear_model_session()

            idx += 1

        if top_fitness_val == 0:
            top_val_final_state = final_state
            top_fitness_indices = []
            top_fitness_indices.append(1)
            best_index = 1

        return top_fitness_val, top_fitness_indices, top_val_final_state, best_index


    def run_eval_on_model(self, model, deme_idx):
        if self.GAME == "Tartarus":
            final_val, final_state = self.eval_tartarus(model, deme_idx)
        elif self.GAME == "Folding":
            final_val, final_state = self.eval_folding(model, deme_idx)

        return final_val, final_state

    def eval_folding(self, model, deme_idx):
        folding = fold.Folding(model)
        data_idx = 0
        cumulative_val = 0

        paths = folding.embed_protein(self.eval_data[0])

        for path, protein in zip(paths, self.eval_data):

            # p_path = np.array([[1,2],[3,4],[5,8]])
            # p_sum = np.sum(p_path != 0, axis=1)


            p_path = path[1:]
            p_sum = np.sum(p_path != 0, axis=1)
            p_non_z = np.nonzero(p_sum)
            path_new = path[np.ix_(p_non_z[0], range(2))]
            final_state = copy.deepcopy(path)



            #individual.behavior_list.append(self.behavior_characteristics(path, sample))
            #individual.fitness_list.append(individual.behavior_list[-1][0])

            if deme_idx == 0:
                cumulative_val += folding.compute_fitness(path, protein)
            elif deme_idx == 1:
                cumulative_val += folding.completeness(path, protein)
            elif deme_idx == 2:
                cumulative_val += folding.compactness(path)
            elif deme_idx == 3:
                cumulative_val += folding.moment_of_inertia(path)

        # characteristics = [
        #     self.compute_fitness(path, protein),
        #     self.completeness(path, protein),
        #     self.compactness(path),
        #     self.moment_of_inertia(path)
        # ]
        # characteristics.extend(self.curvature_and_torsion(path))
        # return np.array(characteristics)

        if cumulative_val != 0:
            final_val = cumulative_val / self.eval_count
        else:
            final_val = 0



        return final_val, final_state

    def eval_tartarus(self, model, deme_idx):
        tartarus = Tartarus()
        board_idx = 0
        cumulative_val = 0
        vId = str(gj.valkyrieID())

        #while board_idx < self.eval_count:
#        print(vId)
        if True:
            move_history = []
            idx = 0

            # Set Board back to same start for each gene in chromosome
            tartarus.set_board(copy.deepcopy(self.eval_data[board_idx]))
            # print("Starting Board________________")
            # print(self.eval_boards[board_idx])
            # print (self.eval_board_start)
            # self.tartarus.print_board()

            print("Valkyrie ID: " + vId + " " + "Starting TF")
            while idx < self.MAX_MOVE_UNITS:


                state = tartarus.get_state_from_sensors()
                encoded = self.encode_move_data(state)
                move_history.append(encoded) #add encoded move to move history

                X = np.reshape(move_history, (1, idx + 1, 24))

                pred = model.predict(X)


                #max = np.argsort(pred)
                max_index = pred.argmax()
                #move_pred = pred[idx]


                if max_index == 0:
                    tartarus.rotate_dozer_direction("LEFT")
                    #print("left")
                elif max_index == 1:
                    tartarus.rotate_dozer_direction("RIGHT")
                    #print("right")
                else:
                    moved = tartarus.try_move_forward()
                    #print("foward")

                idx += 1

            #print(tartarus.get_board())
            if deme_idx == 0:
                corner_val = 2
                edge_val = -1
                cumulative_val = cumulative_val + tartarus.get_final_val(corner_val, edge_val)
                #cumulative_val = cumulative_val + tartarus.get_final_distance_from_center_score()
            elif deme_idx == 1:
                corner_val = 2
                edge_val = 0
                cumulative_val = cumulative_val + tartarus.get_final_val(corner_val, edge_val)
                # #cumulative_val = cumulative_val + tartarus.get_unique_box_score()
                #cumulative_val = cumulative_val + tartarus.get_unique_box_score()
            elif deme_idx ==2:
                cumulative_val = cumulative_val + tartarus.get_final_distance_from_center_score()
            elif deme_idx == 3:
                cumulative_val = cumulative_val + tartarus.get_unique_box_score()
            elif deme_idx == 4:
                cumulative_val = cumulative_val + tartarus.get_inner_box_box_score(2)

            board_idx += 1

        final_state = tartarus.get_final_board_state_as_vector()

        if cumulative_val != 0:
            final_val = cumulative_val / self.eval_count
        else:
            final_val = 0

        return final_val, final_state

    def encode_move_data(self, move_data):
        from keras.utils import to_categorical
        encoded = to_categorical(move_data, num_classes=3)
        encoded = np.reshape(encoded,[1, 24])

        return encoded

    def __init__(self):

        tartarus = Tartarus()
        eval_data = [tartarus.init_new_borad()]
        self.eval_data = copy.deepcopy(eval_data)
        self.eval_count = len(self.eval_data)
        self.MAX_MOVE_UNITS = 80
        self.WEIGHT_NOISE_MU = 0.1
        self.WEIGHT_NOISE_SIGNMA = 0.1
        self.GAME = 'tartarus'


class yggdrasilModel():
    def __init__(self):
        #self.model = self.build_model()
        self.modelBuilt = False
        self.tartarus = Tartarus()
        self.w = [(24,320),
             (80,320),
             (320,),
             (80,3),
             (3)]
        #self.weights = gj.weights_multi(self.w)
        self.evaluate = Evaluate()
        #self.config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, allow_soft_placement=False, device_count = {'CPU': 1})
        #self.graph = tf.Graph()
        #self.session = tf.Session(config=self.config, graph=self.graph)

    def build_model(self):
        # init_seed = evo_seeds[0]
        #init_seed = 3
        #np.random.seed(init_seed) #must set this in addition of seed in kernal_initializer for reproducable results
        #random.seed(init_seed)  # Python
        #set_random_seed(init_seed)  # Tensorflow

        # units = []
        # units.append(80)
        # units.append(80)
        #import sys
        #stdout = sys.stdout
        #sys.stdout = open('/dev/null', 'w')
        input_val = 24
        output_val = 3
        model_layer_units = [80]
        layers = len(model_layer_units)
        layer_count = 0
        test = model_layer_units[layer_count]

        model = Sequential()

        while layer_count < layers - 1:
            model.add(LSTM(model_layer_units[layer_count], input_shape=(None, input_val), return_sequences=True,
                           kernel_initializer=Zeros(),
                           recurrent_initializer=Zeros(), bias_initializer=Zeros()))

            layer_count += 1

        model.add(LSTM(model_layer_units[layer_count], input_shape=(None, input_val), kernel_initializer=Zeros(), recurrent_initializer=Zeros(), bias_initializer=Zeros()))
        model.add(Dense(output_val, kernel_initializer=Zeros(), bias_initializer=Zeros(), activation="softmax"))

        #import tensorflow as tf
        #v = tf.Variable(0, name='my_variable')
        #sess = tf.Session()
        #tf.train.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt')

        return model

def testRun():
    #print(gj.weights())
    #print("bloo")
    # Get a score, damn you!
    #print("Ya'll should piss off!")
    m = gj.weights()
    #print(m)
    mrange = np.arange(-2.5, 2.5, 1.0/float(10))
    mu = 1
    sigma = 0.03
    # Oh, I don't have a norm thing.
    t = norm.pdf(mrange, loc=mu)*1
    mt = np.average(t)
    a = np.sqrt((m - t)**2)
    #print("did it work")
    # Returns a score.  Can we do anything with it?  Ah, ah...
    #print(np.average(a), '\n')
    #print("Don't be a bitch: {}".format(a))
    return (float(np.average(a)))


def orun():
    m = 100000
    y = 0
    for i in range(0,m):
        y += m
    return y

def runParallel():
    from multiprocessing import Pool
    from multiprocessing import Process, Queue
    q = Queue()
    p = Process(target=run, args=(q,))
    p.start()
    p.join()
    a = q.get()
    #a = 0
    #print(a)
    return a
    #with Pool(2) as p:
    #    return p.map(run_model, [])

def runFive():
    return 5.0

loki = yggdrasilModel()
loki.model = loki.build_model()

def run():
    start = time.time()
    vId = str(gj.valkyrieID())
    deme = gj.demeID()
    w = gj.weights_multi(loki.w)
    #print("Valkyrie ID: " + vId + " " + str(w[0][0,0]))
    #print(w)
    #print(loki.w)
    loki.model.set_weights(w)
    #loki.model._make_predict_function()
    #print("okay, eval this shit")
    final_val, final_state = loki.evaluate.eval_tartarus(loki.model, deme)
    #print("Valkyrie ID: " + vId + " " + "TF End")
    #print(final_val)
    #print("Valkyrie ID: " + vId + " " + str(loki.model.get_weights()[0][0,0]))

    #gc.collect()
    #q.put(final_val)
    print("Valkyrie ID: " + vId + " " + "Returning score: " + str(final_val))
    end = time.time()
    p = subprocess.call('echo ' + str(end-start) + '>> /tmp/yggtime', stdout=subprocess.PIPE, shell=True)
    #p.communicate()

    return final_val

def calculateNovelty(a, b, c):
    print("HEY AM I WORKING WHEEEEEEEEEEE", str(b))
    #rL = gj.ref()
    #print("YO FROM WHAT UP: " + str(rL))

    #print(x, y)
    return 12
