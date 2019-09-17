import face_recognition_system as frs

print("1 : Input")
print("2 : Train")
print("3 : Recognize")
print("0 : Quit")

while(True):
	a = int(input("Enter Your Choice : "))

	if a==1:
		Id = input("Enter ID:")
		frs.data(Id)
		print("You have successfully inserted data")

	elif a==2:
		frs.train()
		print("You have trained your model")

	elif a==3:
		frs.recognize()

	elif a==0:
		break

	else:
		print("...Enter Valid Choice...")