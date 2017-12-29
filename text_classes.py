import numpy as np


class Text():
    def __init__(self, text):
        self.in_str = text
        self.len = len(text)
        self.chars = sorted(list(set(text)))
        self.char2id = dict((c, i) for i, c in enumerate(self.chars))
        self.id2char = dict((i, c) for i, c in enumerate(self.chars))
        self.make_in_array()  # Make self.in_array
        
        # Below can be populated with prep_for_training method if use text for training
        self.part_list = None
        self.len_part_list = None
        self.batch_size = None
        self.n_char_per_memory = None
        self.n_char_random_offset_max = None
        
    def __str__(self):
        return self.in_str.replace('\n', '\n\n')

    def set_attr(self, chars, char2id, id2char, n_char_per_memory):
        self.chars = chars
        self.char2id = char2id
        self.id2char = id2char
        self.n_char_per_memory = n_char_per_memory
        self.make_in_array()
    
    def make_in_array(self):
        self.in_array = np.zeros((self.len, len(self.chars)))
        for i, c in enumerate(self.in_str):
            self.in_array[i, self.char2id[c]] = 1
            
    def make_part_list(self):
        n_char_per_part = self.len // self.batch_size
        
        self.part_list = []
        self.len_part_list = []
        for part_id in range(self.batch_size):
            i_start = part_id * n_char_per_part
            if part_id == (self.batch_size - 1):  # Last part should include rest of text
                i_end_exclusive = self.len
            else:
                i_end_exclusive = (part_id + 1) * n_char_per_part
            tp = TextPart(part_id, i_start, i_end_exclusive, self.n_char_per_memory, self.len)
            self.part_list.append(tp)
            self.len_part_list.append(i_end_exclusive - i_start)
            
    def prep_for_training(self, batch_size, n_char_per_memory, n_char_random_offset_max):
        self.batch_size = batch_size
        self.n_char_per_memory = n_char_per_memory
        self.n_char_random_offset_max = None if (n_char_random_offset_max == 0) else n_char_random_offset_max
        self.make_part_list()
        
    def get_batch_array(self, batch_indexes):
        batch_array = np.empty([self.batch_size, self.n_char_per_memory+1, len(self.chars)])        
        for i in range(self.batch_size):
            mark = 0
            for i_start, i_end_exclusive in batch_indexes[i]:
                n = i_end_exclusive - i_start
                batch_array[i, mark: mark+n] = self.in_array[i_start: i_end_exclusive]
                mark += n
        return batch_array
    
    def split_into_XY(self, batch_array):
        X = batch_array[:, : -1, :]  # X.shape = [batch_size, n_char_per_memory, n_char]
        Y = batch_array[:, -1, :] # Y.shape = [batch_size, n_char]
        return X, Y
       
    def get_next_batch(self):
        batch_indexes = []
        for part in self.part_list:
            if self.n_char_random_offset_max is not None:
                offset = np.random.choice(self.n_char_random_offset_max)
            else:
                offset = None
            batch_indexes.append(part.get_next_indexes(offset))     
        batch_array = self.get_batch_array(batch_indexes)
        return self.split_into_XY(batch_array)

    # Below for test
    def get_input_for_generate(self):
        input_ = np.zeros((1, self.n_char_per_memory, len(self.chars)))
        
        i_end_exclusive = len(self.in_str)
        i_start = i_end_exclusive - self.n_char_per_memory
        skip = 0
        
        if i_start < 0:  # If there aren't enough characters to reach n_char_per_memory
            skip = abs(i_start)
            i_start = 0
            
        input_[0, skip:, :] = self.in_array[i_start: i_end_exclusive]
        return input_
    
    def add_char_from_softmax(self, softmax):
        # Pick char based on given probabilities
        char = np.random.choice(self.chars, p=softmax)
        
        # Add to in_str
        self.in_str += char
        
        # Add to in_array
        array = np.zeros((1, len(self.chars)))
        array[0, self.char2id[char]] = 1            
        self.in_array = np.append(self.in_array, array, axis=0)
        
    def reset(self):
        self.in_str = self.in_str[: self.len]
        self.in_array = self.in_array[: self.len]     

class TextPart():
    def __init__(self, part_id, i_start, i_end_exclusive, n_char_per_memory, len_whole):
        self.part_id = part_id
        self.i_start = i_start
        self.i_end_exclusive = i_end_exclusive
        self.n_char_per_memory = n_char_per_memory
        self.len_whole = len_whole
        self.i_current = i_start
        
    def get_next_indexes(self, offset):
        # Start index
        i_start = self.i_current
        if offset is not None:
            i_start = (self.i_current - offset) % self.len_whole
        
        # End index
        i_end_exclusive = (i_start + self.n_char_per_memory + 1) % self.len_whole  # +1 for last output
        
        # Split if end of text is in middle
        if i_end_exclusive <= i_start:
            output = [(i_start, self.len_whole), (0, i_end_exclusive)]
        else:
            output = [(i_start, i_end_exclusive)]
        
        self.update_i_current(i_end_exclusive)
        return output
    
    def update_i_current(self, i_end_exclusive):
        self.i_current = (i_end_exclusive - 2) % self.len_whole  # -1 for last output
        
        # If beyond allocated part, go to head
        if self.i_end_exclusive == self.len_whole:  # Last one is tricky
            if self.i_current < self.i_start:
                self.i_current = self.i_start
        else:
            if self.i_end_exclusive <= self.i_current:  
                self.i_current = self.i_start   
                