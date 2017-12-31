import bingo_mod

def main():
	mat = bingo_mod.initMatrix(8)
	bingo_mod.printMatrix(mat)

	bingo_numbers=[]
	while True:
		number=raw_input('Input bingo number: ')

		try:
			number=int(number)	# check if integer
			bingo_mod.updateMatrix(mat, number)	# update bingo matrix
			bingo_mod.printMatrix(mat)	# print matrix
			bingo_numbers.append(number)	# log bingo numbers
			print

		except:
			print ("Number not INTEGER")


		print ("Bingo numbers: ", bingo_numbers)

if __name__=="__main__":
	main()
