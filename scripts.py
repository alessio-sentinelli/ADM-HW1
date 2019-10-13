#################################### Problem 1 #####################################


#____________________________________Introduction____________________________________


#___Say "Hello, World!" With Python___
my_name = "Hello, World!"
print(my_name)


#___Python If-Else___
import math
import os
import random
import re
import sys
if __name__ == '__main__':
    n = int(input().strip())
    if n % 2 != 0 or 5 < n < 21:
        print('Weird')
    else:
        print('Not Weird')


#___Arithmetic Operators___
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a + b)
    print(a - b)
    print(a * b)
    

#___Python: Division___
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a // b)
    print(a / b)


#___Loops___
if __name__ == '__main__':
    n = int(input())
    for i in range(0,n):
        print(i**2)


#___Write a function___
def is_leap(year):
    leap = False
    if year % 4 == 0:
        leap = True
        if year % 100 == 0 and year % 400 != 0:
            leap = False
    return leap
year = int(input())
print(is_leap(year))


#___Print Function___
if __name__ == '__main__':
    n = int(input())
    for i in range(1,n+1):
        print(i, end = '')



#__________________________________Basic data Type__________________________________

#___List Comprehensions___
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    out =  []
    for i in range(x+1):
        for j in range(y+1):
            for k in range(z+1):
                if i + j + k != n:
                    out.append([i, j, k])
    print(out)


#___Find the Runner-Up Score!___
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    run = sorted(arr, reverse = True)
    i = 0
    while run[i] == run[i+1]:
        i = i + 1
    print(run[i+1])


#___Nested Lists___
if __name__ == '__main__':
    stud_name = []
    stud_score = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        stud_name.append(name)
        stud_score.append(score)
    sort_score = sorted(stud_score)
    smaller = sort_score[0]
    sort_score.remove(smaller)
    count = 0
    while count == 0:
        if sort_score[0] == smaller:
            sort_score.remove(smaller)
        else:
            count = 1
    value = []
    for i in range (stud_score.count(sort_score[0])):
        position = stud_score.index(sort_score[0])
        stud_score[position] = 0
        value.append(stud_name[position])
    for i in range (len(value)-1, -1, -1):
        print(value[i])


#___Finding the percentage___
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    print(mean.query_name)


#___Lists___
if __name__ == '__main__':
    N = int(input())
    L = []
    for i in range(0, N):
        tokens = input().split()
        if tokens[0] == 'insert':
            L.insert(int(tokens[1]), int(tokens[2]))
        elif tokens[0] == 'print':
            print (L)
        elif tokens[0] == 'remove':
            L.remove(int(tokens[1]))
        elif tokens[0] == 'append':
            L.append(int(tokens[1]))
        elif tokens[0] == 'sort':
            L.sort()
        elif tokens[0] == 'pop':
            L.pop()
        elif tokens[0] == 'reverse':
            L.reverse()


#___Tuples___
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    print(hash(tuple(integer_list)))


#_____________________________________Strings________________________________________

#___sWAP cASE___
def swap_case(s):
    return s.swapcase()
if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)


#___String Split and Join___
def split_and_join(line):
    Line = []
    for i in range(len(line)):
        if line[i] == " ":
            Line.append("-")
        else:
            Line.append(line[i])
    return "".join(Line)
if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)


#___What's Your Name?___
def print_full_name(a, b):
    print("Hello " + str(a) + " " + str( b) +"! You just delved into python.")
if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)


#___Mutations___
def mutate_string(string, position, character):
    change = list(string)
    change[position] = character
    return "".join(change)
if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)


#___Find a string___
def count_substring(string, sub_string):
    count = 0
    for i in range(len(string)-len(sub_string) + 1):
        #print(string[i:i+len(sub_string)], sub_string)
        if string[i:i+len(sub_string)] == sub_string:
            count += 1
    return count
if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    count = count_substring(string, sub_string)
    print(count)


#___String Validators___
if __name__ == '__main__':
    s = input()
    print(any([val.isalnum() for val in s]))
    print(any([val.isalpha() for val in s]))
    print(any([val.isdigit() for val in s]))
    print(any([val.islower() for val in s]))
    print(any([val.isupper() for val in s]))
    

#___Text Alignment___
thickness = int(input()) #This must be an odd number
c = 'H'
#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))
#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    
#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    
#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-
i-1)).ljust(thickness)).rjust(thickness*6))


#___Text Wrap___
import textwrap
def wrap(string, max_width):
    line = list(string)
    count = 0
    for i in range(max_width,len(string)-len(string)%max_width+1,max_width):
        line.insert(i+count, '\n')
        count += 1 
    return ''.join(line)
if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)


#___Designer Door Mat___
if __name__ == '__main__':
    N, M = map(int, input().split())
top = [".","|","."]
TOP_part=[]
for i in range(0, int(N/2)):
    line = ["-" for k in range(0, int(M/2) - i*3-1)]
    line_ = ["-" for k in range(0, int(M/2) - i*3-1)]
    for m in range(1,i*2 + 2):
        line += top
    line += line_
    TOP_part.append(line)
line = ["-" for i in range(int(M/2) - 3)]
TOP_part.append([])   
TOP_part[-1] = TOP_part[-1] + line + ["WELCOME"] + line
for i in range(0, int(N/2)):
    TOP_part.append(TOP_part[int(N/2) - i -1 ])
for i in range(N):
    print(''.join(TOP_part[i]))



#___String Formatting___
def print_formatted(number):
    max_width = len(bin(number)[2:])
    for i in range(1,number+1):
        print(' '.join(map(lambda x: x.rjust(max_width), [str(i), oct(i)[2:], hex(i)[2:].upper(), bin(i)[2:]])))


#___Capitalize!___
def solve(s):
    words = s.split(' ')
    Words = [word.capitalize() for word in words]
    return ' '.join(Words)


#___The Minion Game___
def minion_game(string):
    # your code goes here
    Stuart, Kevin = 0, 0
    n = len(string)
    Vowels = ['A', 'E', 'I', 'O', 'U']
    for i in range(n):
        if string[i] in Vowels:
            Kevin += n - i
        else:
            Stuart += n - i
    if Kevin < Stuart:
        print('Stuart', Stuart)
    elif Kevin > Stuart:
        print('Kevin', Kevin)
    else:
        print('Draw')


#___Alphabet Rangoli___
#I looked at the Solution
import string
def print_rangoli(size):
    a = string.ascii_lowercase
    r = []
    for i in range(size):
        s = "-".join(a[i:size])
        r.append((s[::-1]+s[1:]).center(4*size-3, "-"))
    print('\n'.join(r[:0:-1]+r)) 
    return

if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)


#___Merge the Tools!___
def merge_the_tools(string, k):
    val = int(len(string) / k)
    for i in range(val):
        t = string[i*k : (i+1)*k]
        u = ""
        for j in t:
            if (j not in u):
                u += j
        print(u)
    return 
if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)



#________________________________________Sets________________________________________

#___Introduction to Sets___
def average(array):
    Array = set(array)
    return sum(Array)/len(Array)


#___No Idea!___
newlist = list(map(int, input().split()))
my = map(int, input().split())
A = set(map(int, input().split()))
B = set(map(int, input().split()))
n, m = newlist[0], newlist[1]
happy = 0
for i in my:
    if i in A:
        happy += 1
    if i in B:
        happy -= 1
print(happy)


#___Summetric Difference___
M = input()
Mset = set(input().split())
N = input()
Nset = set(input().split())
MinterN = Mset.intersection(Nset)
diff = Mset.difference(MinterN).union(Nset.difference(MinterN))
newlis = sorted(list(map(int, diff)))
for i in newlis:
    print(i, sep = '\n')


#___Set.add()___
n = int(input())
country = set()
for i in range(n):
    country.add(input())
print(len(country))



#___Set.discart(), ,remove() & .pop()___
n = int(input())
s = set(map(int, input().split()))
N = int(input())
for i in range(N):
    com = input().split()
    if com[0] == "remove":
        s.remove(int(com[1]))
    elif com[0] == "discard":
        s.discard(int(com[1]))
    else:
        s.pop()
print (sum(list(s)))



#___Set.union() Operation___
n = int(input())
En = set(map(int, input().split()))
b = int(input())
Fr = set(map(int, input().split()))
E_F = En.union(Fr)
print(len(E_F))


#___Set.intersection() Operation___
n = int(input())
En = set(map(int, input().split()))
b = int(input())
Fr = set(map(int, input().split()))
E_F = En.intersection(Fr)
print(len(E_F))



#___Set.difference() Operation___
n = int(input())
En = set(map(int, input().split()))
b = int(input())
Fr = set(map(int, input().split()))
E_F = En.difference(Fr)
print(len(E_F))



#___Set-symmetric_difference() Operation___
n = int(input())
En = set(map(int, input().split()))
b = int(input())
Fr = set(map(int, input().split()))
E_F = En.symmetric_difference(Fr)
print(len(E_F))



#___Set Mutation___
length = int(input())
A = set(map(int, input().split()))
N = int(input())
for i in range(N):
    (op, l_op) = input().split()
    s2 = set(map(int, input().split()))
    if op == 'intersection_update':
        A &= s2
    elif op == 'update':
        A |= s2
    elif op == 'symmetric_difference_update':
        A ^= s2
    elif op == 'difference_update':
        A -= s2
print (sum(A))



#___The Captain's Room___
# I looked at the solutions. My code (indented) had time problems for large K and group
K = int(input())
group = list(map(int, input().split()))
room = set(group)
print (int((sum(room) * K - sum(group)) / (K - 1)))
'''
for i in room:
    if group.count(i) == 1:
        print (i)
'''


#___Check Subset___
n = int(input())
for i in range(n):
    a = int(input())
    A = set(map(int, input().split()))
    b = int(input())
    B = set(map(int, input().split()))
    if A.intersection(B) == A:
        print('True')
    else:
        print('False')



#___Check Strict Superset___
A = set(map(int, input().split()))
n = int(input())
no = 0
for i in range(n):
    B = set(map(int, input().split()))
    if A.issuperset(B) != True or (len(B) == len(A)):
        print('False')
        no = 1
        break
if no == 0 :
    print('True')



#___________________________________Collections______________________________________

#___collections.Counter()___
from collections import Counter
X = int(input())
N = list(map(int, input().split()))
x_i = int(input())
count = Counter(N)
#print(N, count)
soldi = 0
for i in range(x_i):
    size, money = map(int, input().split())
    if count[size] > 0:
        count[size] -= 1
        soldi += money
print(soldi)


#___DefaultDict Tutorial___
from collections import defaultdict
n, m = map(int, input().split())
A = defaultdict(list)
for i in range(0, n):
    A[input()].append(i + 1)

for i in range(0, m):
    value = input()
    if len(A[value]) > 0:
        print ((" ").join(str(c) for c in A[value]))
    else:
        print (-1)



#___collections.namedtuple()___
#I looked at the solution for take the value <student> inside the for loop
import collections
N = int(input())
columns_name = ','.join(input().split())
Name = collections.namedtuple('Student', columns_name)
Sum = 0
for i in range(N):
    line = input().split()
    student = Name(*line)
    Sum += int(student.MARKS)
print (Sum / N)



#___collections.OrderedDict()___
#I looked at the solution
import collections 
N = int(input())
d = collections.OrderedDict()
for i in range(N):
    item = input().split()
    item_price= int(item[-1])
    item_name= " ".join(item[:-1])
    if d.get(item_name):
        d[item_name] += item_price
    else:
        d[item_name] = item_price
for i in d.keys():
    print (i, d[i])



#___Word Order___
import collections 
N = int(input())
d = collections.OrderedDict()
for i in range(N):
    item = input().split()
    item_name = " ".join(item)
    if d.get(item_name):
        d[item_name] += 1
    else:
        d[item_name] = 1
print(len(d.keys()))
for i in d.keys():
    print (d[i], end = " ")



#___collections.deque()___
from collections import deque
d = deque()
N = int(input())
for i in range(N):
    args = input().strip().split()
    if (args[0] == 'append'):
        d.append(args[1])
    elif (args[0] == 'pop'):
        d.pop()
    elif (args[0] == 'popleft'):
        d.popleft()
    elif (args[0] == 'appendleft'):
        d.appendleft(args[1])
print (' '.join(d))



#___Company Logo___
import math
import os
import random
import re
import sys
import collections
if __name__ == '__main__':
    s = sorted(list(input().strip()))
    s_counter = collections.Counter(s).most_common()
    for i in range(3):
        print(s_counter[i][0], s_counter[i][1])


#___Piling Up!___
from collections import deque
N = int(input())
for i in range(N):
    n = int(input())
    side_length = deque(list(map(int,input().split())))
    good = 1
    value = 1
    for i in range(n-1):
        if side_length[-1] >= side_length[-2]:
            side_length.pop()
        elif  side_length[0] >= side_length[1]:
            side_length.popleft()
        else:
            value = 0
            good = 0
    if value ==1:
        print('Yes')
    else:
        print('No')



#______________________________________Date Time____________________________________

#___Calendar Module___
# I looked at the solution
import calendar
date = input().strip().split()
days = list(calendar.day_name)
print (days[calendar.weekday(int(date[2]), int(date[0]), int(date[1]))].upper() )



#___Time Delta___
# I looked at the solution
import math
import os
import random
import re
import sys
from datetime import datetime
def time_delta(t1, t2):
    time_format = '%a %d %b %Y %H:%M:%S %z'
    t1 = datetime.strptime(t1, time_format)
    t2 = datetime.strptime(t2, time_format)
    return str(int(abs((t1-t2).total_seconds())))   
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()



#_______________________________________Exceptions___________________________________
n = int(input())
for i in range(n):
    try:
        a = list(map(int, input().split()))
        print(a[0]//a[1])
    except ZeroDivisionError as e:
        print ("Error Code:",e)
    except ValueError as e:
        print ("Error Code:",e)




#______________________________________Builts-In____________________________________

#___Zipped___
N, X = map(int, input().split())
Table = list()
for x in range(X):
    Table.append(map(float, input().split()))
New_Tatle = zip(*Table)
for i in New_Tatle:
    print(sum(i)/X)



#___python sort sotr___
import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())

New_arr = sorted(arr, key=lambda x:x[k])
for i in range(n):  
    print(" ".join(str(x) for x in New_arr[i])) 



#___ginorts___
#I looked at the solution
print(*sorted(input(), key=lambda c: (c.isdigit() - c.islower(), c in '02468', c)), sep='')




#_____________________________Regex and Parsing challenges__________________________

#___Re.split()___
# I looked at the solutions
regex_pattern = r"[.,]+"	# Do not delete 'r'.
import re
print("\n".join(re.split(regex_pattern, input())))

#I will do the other exercises on Regex, XML, Closures and Decoration later

#_________________________________Pythonn Functional________________________________

#___Map and Lambda Function___
cube = lambda x: x**3 
def fibonacci(n):
    s = [0, 1]
    for i in range(n):
        s.append(s[-1] + s[-2])
    return s[0:n]
if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))



#_____________________________________Numpy__________________________________________

#___Arrays___
import numpy
def arrays(arr):
    return numpy.array(arr, dtype = float)[::-1]
arr = input().strip().split(' ')
result = arrays(arr)
print(result)


#___Shape and Reshape___
import numpy
arr = list(map(int, input().split()))
Arr = numpy.array(arr, dtype = int)
print(numpy.reshape(Arr, (3, 3)))



#___Transpose and Flatten___
import numpy
n_m = list(map(int, input().split()))
N = int(n_m[0])
M = int(n_m[1])
V = []
for i in range(N):
    V.append(list(map(int, input().split())))
    vec = numpy.array(V, dtype = int)
print(numpy.transpose(vec))
print(vec.flatten())


#___Concatenate___
import numpy
N, M, P = map(int, input().split())

NxP = [list(map(int, input().split())) for i in range(N)]
MxP = [list(map(int, input().split())) for i in range(M)]

array_1 = numpy.array(NxP)
array_2 = numpy.array(MxP)

print(numpy.concatenate((array_1, array_2), axis = 0))


#___Zeros and Ones___
import numpy
value = list(map(int,input().split()))
matrix_0 = numpy.zeros((value), dtype = int)
matrix_1 = numpy.ones((value), dtype = int)
print(matrix_0)
print(matrix_1)


#___Eye and Identity___
import numpy
# I had to use the solutions only for the following line, otherwise it gave wrong code
numpy.set_printoptions(sign=' ')
N, M = map(int, input().split())
print(numpy.eye(N, M))



#___Array Mathematics___
import numpy
N, M = map(int, input().split())
A = numpy.array([list(map(int, input().split())) for n in range(N)])
B = numpy.array([list(map(int, input().split())) for n in range(N)])
print (A + B)
print (A - B)
print (A * B)
print (A // B)
print (A % B)
print (A ** B)



#___Floor, Ceil and Rint___
import numpy
numpy.set_printoptions(sign=' ')
array= numpy.array( list(map(float, input().split())))
print (numpy.floor(array))
print (numpy.ceil(array))
print (numpy.rint(array))



#___Sum and Prod___
import numpy as np
N, M = map(int, input().split())
matrix = []
for i in range(N):
    matrix.append(list(map(int, input().split())))
Matrix = np.array(matrix)
sum_0 = np.sum(Matrix, axis = 0)
print (np.prod(sum_0))



#___Min and Max___
import numpy as np
N, M = map(int, input().split())
matrix = []
for i in range(N):
    matrix.append(list(map(int, input().split())))
Matrix = np.array(matrix)
min_1 = np.min(Matrix, axis = 1)
print(np.max(min_1))


#___Mean, Var, and Std___
import numpy as np
np.set_printoptions(sign=' ')
np.set_printoptions(legacy='1.13')      #I need to put this line if I want have the same output of hackerrank
N, M = map(int, input().split())
matrix = []
for i in range(N):
    matrix.append(list(map(int, input().split())))
Matrix = np.array(matrix)
print(np.mean(Matrix, axis = 1))
print(np.var(Matrix, axis = 0))
print(np.std(Matrix))   


#___Dot and Cross___
import numpy as np
N = int(input())
matrix = [[], []]       #matrix contain A and B
for j in range(2):
    for i in range(N):
        matrix[j].append(list(map(int, input().split())))
Matrix = np.array(matrix)
print(np.dot(Matrix[0],Matrix[1]))


#___Inner and Outer___
import numpy
A = numpy.array(list(map(int, input().split())))
B = numpy.array(list(map(int, input().split())))
print (numpy.inner(A, B))
print (numpy.outer(A, B))



#___Polynomials___
import numpy as np
print (np.polyval(np.array(list(map(float, input().split()))), float(input()))) 



#___Linear Algebra___
import numpy as np
np.set_printoptions(legacy='1.13')
N = int(input())
A = [list(map(float, input().split())) for i in range(N)]
A = np.matrix(A)
print(np.linalg.det(A))

















#################################### Problem 2 #####################################


#___Birthday Cake Candles___
import math
import os
import random
import re
import sys
def birthdayCakeCandles(ar):
    Max = max(ar)
    count = 0
    for i in range(len(ar)):
        if ar[i] == Max:
            count += 1
    return count
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    ar_count = int(input())

    ar = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(ar)

    fptr.write(str(result) + '\n')

    fptr.close()




#___Kangaroo___
import math
import os
import random
import re
import sys
def kangaroo(x1, v1, x2, v2):
    #using formula in first example
    if x1 < x2 and v1 < v2: 
        return 'NO'
    for i in range(10**5):      
        if x1 + v1 == x2 + v2:
            return 'YES'
        x1 += v1    #kangaroos jump
        x2 += v2
    return 'NO'
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()






#___Viral Advertising___
import math
import os
import random
import re
import sys
def viralAdvertising(n):
    day_share = 5
    day_like = int(day_share/2)
    total = day_like
    for i in range(n-1):
        day_share = day_like*3
        day_like = int(day_share/2)
        total += day_like
    return total
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()





#___Recursive Digit Sum___
import math
import os
import random
import re
import sys

# Complete the superDigit function below.
def superDigit(n, k):
    first_step = sum(map(int, list(n)))        
    p = str(first_step)*k                       
    P = list(map(int, list(p)))
    while len(p) == 1:
        return P[0]
    return superDigit(P, 1)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = int(nk[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()




#___Insertion Sort - Part 1___
import math
import os
import random
import re
import sys
def insertionSort1(n, arr):
    Sort = 0
    for i in range(n-1):
        if arr[i] > arr[i+1]:
            Value = arr[i+1]
            index_Value = i+1
    while Sort == 0:
        if Value < arr[index_Value-1]:
            arr[index_Value] =  arr[index_Value-1]
            index_Value += -1 
            print (" ".join(str(f) for f in arr))
            if index_Value < 1:
                arr[index_Value] =  Value
                Sort = 1
        else:    
            arr[index_Value] =  Value
            Sort = 1
    print (" ".join(str(f) for f in arr))
    return
if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)



#___Insertion Sort - Part 2___
import math
import os
import random
import re
import sys
def insertionSort2(n, arr):
    for i in range(1, n):
        k = arr[i]
        j = i
        while j > 0 and k < arr[j-1]:
            arr[j] = arr[j-1]
            j += -1
        arr[j] = k
        print (' '.join(str(j) for j in arr))
    return
if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)





















