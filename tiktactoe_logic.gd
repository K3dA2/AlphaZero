extends GridContainer

# 2D array to represent the board (3x3 grid)
var board = [[null, null, null], [null, null, null], [null, null, null]]
var current_player = "X"
var game_over = false
var cpu_playing = true  # Set to true for Player vs CPU mode

# WebSocket client to communicate with Python server
var client = WebSocketPeer.new()
var url = "ws://localhost:5000"

func _ready():
	# Connect to the server
	var err = client.connect_to_url(url)
	if err != OK:
		print("Unable to connect to Python server")
		set_process(false)
		
	for i in range(9):
		var button_name = "Button%d" % (i + 1)
		var button = get_node(button_name)
		button.pressed.connect(_on_Button_pressed.bind(i))

func _process(delta):
	client.poll()  # Poll WebSocket for incoming messages
	
	# get_ready_state() tells you what state the socket is in.
	var state = client.get_ready_state()
	# WebSocketPeer.STATE_OPEN means the socket is connected and ready
	# to send and receive data.
	if state == WebSocketPeer.STATE_OPEN:
		while client.get_available_packet_count():
			var action_str = client.get_packet().get_string_from_utf8()
			print("Got data from server: ", action_str)
			var action = int(action_str)  # Convert received action to an integer
			make_cpu_move(action)

	# WebSocketPeer.STATE_CLOSING means the socket is closing.
	# It is important to keep polling for a clean close.
	elif state == WebSocketPeer.STATE_CLOSING:
		pass

# Called when a button is clicked
func _on_Button_pressed(button_id):
	if game_over:
		return # Stop further clicks if the game is over
	
	var row = int(button_id / 3)
	
	var col = button_id % 3

	# Check if the cell is empty before placing a move
	if board[row][col] == null:
		# Update board with the current player's move
		board[row][col] = current_player
		# Get the button node by dynamically building its name
		var button_name = "Button%d" % (button_id + 1)  # button1, button2, etc.
		get_node(button_name).text = current_player  # Set the text to X or O
		# Check for a winner or a draw
		if check_winner():
			print("%s wins!" % current_player)
			$"../Label".text = current_player + " wins"
			game_over = true
		elif check_draw():
			print("It's a draw!")
			$"../Label".text = "It's a draw!"
			game_over = true
		else:
			switch_player()
			if cpu_playing:
				# Send board state to the server for CPU move
				send_to_server()

# Switch the current player
func switch_player():
	current_player = "O" if current_player == "X" else "X"

# Send board state to the server
func send_to_server():
	var flat_board = []
	for row in board:
		for cell in row:
			flat_board.append(cell if cell != null else " ")  # Use " " for empty cells
	var msg = JSON.stringify(flat_board)
	client.put_packet(msg.to_utf8_buffer())

# Called when data is received from the server (CPU's move)
func _on_data_received():
	var action_str = client.get_packet().get_string_from_utf8()
	var action = int(action_str)  # Convert received action to an integer
	print("action: ",action_str)
	make_cpu_move(action)
	
# CPU makes a move based on server response
func make_cpu_move(action):
	var row = int(action / 3)
	var col = action % 3

	if board[row][col] == null:
		board[row][col] = current_player
		var button_name = "Button%d" % (action + 1)
		get_node(button_name).text = current_player
		
		# Check for winner or draw after CPU move
		if check_winner():
			print("%s wins!" % current_player)
			$"../Label".text = current_player + " wins!"
			game_over = true
		elif check_draw():
			print("It's a draw!")
			$"../Label".text = "It's a draw!"
			game_over = true
		else:
			switch_player()  # Switch back to player

# Check if there's a winner
func check_winner() -> bool:
	# Check rows
	for row in range(3):
		if board[row][0] == current_player and board[row][1] == current_player and board[row][2] == current_player:
			return true

	# Check columns
	for col in range(3):
		if board[0][col] == current_player and board[1][col] == current_player and board[2][col] == current_player:
			return true

	# Check diagonals
	if board[0][0] == current_player and board[1][1] == current_player and board[2][2] == current_player:
		return true
	if board[0][2] == current_player and board[1][1] == current_player and board[2][0] == current_player:
		return true

	# No winner found
	return false

# Check if the board is full (i.e., a draw)
func check_draw() -> bool:
	for row in board:
		if null in row:
			return false
	return true

# Reset the game to the initial state
func reset_game():
	board = [[null, null, null], [null, null, null], [null, null, null]]
	game_over = false
	current_player = "X"
	# Clear button texts
	for i in range(9):
		var button_name = "Button%d" % (i + 1)
		get_node(button_name).text = ""
	$"../Label".text = ""

# Connect each button to this function in the Godot editor to call _on_Button_pressed when clicked


func _on_button_pressed():
	reset_game()


func _on_button_2_pressed():
	pass # Replace with function body.
