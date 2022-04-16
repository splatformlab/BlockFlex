import threading
import lstm_sz
import lstm_bw


def main():
    thr = []
    t_thread = threading.Thread(target=lstm_bw.main, args=())
    thr.append(t_thread)
    t_thread = threading.Thread(target=lstm_sz.main, args=())
    thr.append(t_thread)
    for thread in thr: 
        thread.start()
    for thread in thr:
        thread.join()

if __name__ == "__main__":
    main()
