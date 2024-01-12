import socket
import time

Game="A5x5" # "A5x5" "A6x6" "G7x7" "G9x9"

def start_server(host='localhost', port=12345):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(2)

    print("Waiting for two agents to connect...")
    agent1, addr1 = server_socket.accept()
    print("Agent 1 connected from", addr1)
    bs=b'AG1 '+Game.encode()
    agent1.sendall(bs)

    agent2, addr2 = server_socket.accept()
    print("Agent 2 connected from", addr2)
    bs=b'AG2 '+Game.encode()
    agent2.sendall(bs)    

    agents = [agent1, agent2]
    current_agent = 0

    jog=0
    
    while True:
        try:
            data = agents[current_agent].recv(1024).decode()
            if not data:
                break

            # Process the move (example: "MOVE X,Y")
            print(current_agent, " -> ",data)
            jog = jog+1
            
            if jog==100: 
                agents[current_agent].sendall(b'END 0 10 10')
                agents[1-current_agent].sendall(b'END 0 10 10')
                break
            
            if is_valid_move(data):
                agents[current_agent].sendall(b'VALID')
                agents[1-current_agent].sendall(data.encode())
            else:
                agents[current_agent].sendall(b'INVALID')

            # Switch to the other agent
            current_agent = 1-current_agent
            time.sleep(1)

        except Exception as e:
            print("Error:", e)
            break

    print("\n-----------------\nGAME END\n-----------------\n")
    time.sleep(1)
    agent1.close()
    agent2.close()
    server_socket.close()

def is_valid_move(move):
    # Implement the logic to check if the move is valid
    # G6's agents do not need this since they can only play legal moves (checked internally)
    return True

if __name__ == "__main__":
    start_server()
