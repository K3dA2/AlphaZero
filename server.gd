extends Node

var client = WebSocketPeer.new()
var url = "ws://localhost:5000"
# Called when the node enters the scene tree for the first time.

func _ready():
	#client.connect("data_recieved",data_recieved())
	client.connect("data_recieved",Callable(self,"data_recieved"))
	var err = client.connect_to_url(url)
	if err != OK:
		set_process(false)
		print("Unable to connect")


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	client.poll()
	
	# get_ready_state() tells you what state the socket is in.
	var state = client.get_ready_state()
	# WebSocketPeer.STATE_OPEN means the socket is connected and ready
	# to send and receive data.
	if state == WebSocketPeer.STATE_OPEN:
		while client.get_available_packet_count():
			var payload = client.get_packet().get_string_from_utf8()
			print("Got data from server: ", payload)
			$debug.text += "Got message: " + payload

	# WebSocketPeer.STATE_CLOSING means the socket is closing.
	# It is important to keep polling for a clean close.
	elif state == WebSocketPeer.STATE_CLOSING:
		pass
	
func data_recieved():
	print("active")
	var payload = client.get_packet().get_string_from_utf8()
	$debug.text += "Got message: " + payload
	
func send(msg):
	client.put_packet(JSON.stringify(msg).to_utf8_buffer())
	


func _on_button_pressed():
	send($LineEdit.text)
