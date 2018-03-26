import tempfile
import time
import random
import os
from tqdm import tqdm

def shuffle_big_files(filenames, line_cnt = None, chunk_size = 1000000):
    '''
    Provide a method to random shuffle big files, the content of the files should be seperated by line.
    Args:
        - filenames: a list of filenames which you want to shuffle with
        - line_cnt: the number of lines of each file, the line_cnt must be same for all the files, if line_cnt is None,
                    we will calculate line number on line, which is much slower
        - chunk_size: size of memory cached data size, it depends on your RAM and length of lines. Use small number if your memory
                    is small or the line of the file will be very long. Otherwise use bigger number
    Return:
        - new list of filenames ends with .shuffle
    '''
    s = time.time()
    
    if not isinstance(filenames, list):
        filenames = [filenames]
    
    for fname in filenames:
        if not os.path.exists(fname):
            raise ArgumentError("file {0} not exists".format(fname))
    
    if line_cnt is None:
        line_cnt = sum(1 for line in open(filenames[0]))
    
    print("shuffling {0} lines of {1} files...".format(line_cnt,len(filenames)))
    
    def read_files(filenames):
        '''
        Read lines from multi files, join by \x01
        '''
        fps = [open(fname) for fname in filenames]
        cnt = 0
        while True:
            if cnt == line_cnt:
                break
            tokens = []
            for f in fps:
                line = f.readline().rstrip('\n')
                tokens.append(line)
            yield '\x01'.join(tokens)
            cnt += 1
        for f in fps:
            f.close()

    def random_permutation(lines):
        for i, n in enumerate(lines):
            r = random.randint(0, i)
            lines[i] = lines[r]
            lines[r] = n
        return lines

    def write_to_file(chunk, fp):
        for line in chunk:
            fp.write(line + '\n')
    
    def shuffle_and_write(chunk, dirname):
        print("readed {0} lines, shuffle and write to temp file".format(chunk_size))
        chunk = random_permutation(chunk)
        tmp_fp = tempfile.TemporaryFile(mode="w+",dir=dirname)
        write_to_file(chunk,tmp_fp)
        tmp_fp.seek(0)
        return tmp_fp
        
    # create a list of temp files, and random insert the lines into these temp files
    def split_file_into_chunks(filenames):
        dirname = os.path.dirname(filenames[0])
        fps = []
        chunk = []
        print("read lines from files...")
        for line in tqdm(read_files(filenames)):
            chunk.append(line)
            if len(chunk) == chunk_size:
                tmp_fp = shuffle_and_write(chunk, dirname)
                fps.append(tmp_fp)
                chunk = []
        if len(chunk) > 0:
            tmp_fp = shuffle_and_write(chunk, dirname)
            fps.append(tmp_fp)
        return fps
    
       
    def read_chunk_from_temp_files(fps):
        small_chunk = []

        def read_small_chunk():
            empty_fps = []
            for i,f in enumerate(fps):
                line = f.readline()
                if not line:
                    empty_fps.append(i)
                    continue
                small_chunk.append(line.rstrip('\n'))
            return empty_fps
        
        while len(fps) > 0:
            empty_fps = read_small_chunk()
            fps = [f for i,f in enumerate(fps) if i not in empty_fps]
            if len(small_chunk) >= chunk_size:
                yield small_chunk
                chunk = []
        if len(small_chunk) != 0:
            yield small_chunk
        
    # merge
    def merge_chunks_into_files(fps):
        print("merge temp files into final shuffle files...")
        out_files = [open(fname + '.shuffle','w') for fname in filenames]
        for chunk in read_chunk_from_temp_files(fps):
            chunk = random_permutation(chunk)
            for line in chunk:
                tokens = line.split('\x01')
                if len(tokens) != len(filenames):
                    continue
                for i,token in enumerate(tokens):
                    out_files[i].write(token + '\n')
        for f in out_files:
            f.close()
            
    fps = split_file_into_chunks(filenames)
    merge_chunks_into_files(fps)
    e = time.time()
    print("finish shuffling in {0} secs".format(e-s))
    return [fname + '.shuffle' for fname in filenames]