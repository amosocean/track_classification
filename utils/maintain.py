import numpy as np
from threading import Thread
import time
from queue import Queue
class dummy_server:
    def __init__(self) -> None:
        self.count = np.int32(1) #总包数号
        self.CurMsg = np.int32(0) #最后一个flag
        self.trackID = np.random.randint(10000,10002) 
        self.posNum = np.int32(50) #航迹有效点数
        self.track = np.ones([6,50],dtype=np.float64)
    
    def dummy_track(self):
        while True:
            time.sleep(1)
            Pack_buffer.put({"count":self.count,"flag":self.CurMsg,"trackID":self.trackID,"track":self.track})
            self.__init__()
        #return {"count":self.count,"flag":self.CurMsg,"trackID":self.trackId,"track":self.track}

def classify(track_split_list):
    print(len(track_split_list))
    return [2,3,4]

def main():
    global Pack_buffer
    Pack_buffer = Queue()
    local_dict = {}
    server = dummy_server()
    server_td = Thread(target=server.dummy_track)
    server_td.start()
    while True:
        if not Pack_buffer.empty():
            current_split = Pack_buffer.get()
            track_id= current_split["trackID"]
            if local_dict.get(track_id) is None:
                local_dict[track_id] = []
            if current_split["flag"]:
                pass  #处理最后一个不完整分片
            local_dict[track_id].append(current_split["track"])
            result = classify(local_dict[track_id])
            if current_split["flag"]:
                local_dict.pop(track_id) #轨迹split已经全部识别删除这条轨迹
            print(result)
            print("#############")
            time.sleep(0.1)
    #track_dicts = [x for x in range(10) server.dummy_track()]

if __name__ == "__main__":
    main()