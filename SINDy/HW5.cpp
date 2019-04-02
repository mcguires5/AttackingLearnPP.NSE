/*
* @Author: delengowski
* @Date:   2019-03-27 21:08:04
* @Last Modified by:   delengowski
* @Last Modified time: 2019-03-27 22:08:46
*/
// PROBLEM 1
template <class Item>
Item minimal(const Item data[], SizeType n)
{
  item result;
  int i;
  for(i = 0; i < n, i++)
  {
  	if(i == 0) // store first value of array
  	{
  		result = data[i];
  	}
  	else // for all steps after, check if next element is smaller
  	{
  		if(result > data[i])
  		{
  			result = data[i];
  		}
  	}
  }
  return (result);
}

// PROBLEM 2 A
// Answer A, you can only assign output iterators, you cannot access them as a rvalue.

// PROBLEM 2 B
// Looking at the documentation of the multiset documentation for STL,
// it appears that the insert method will just insert the value. The 
// caveat is that the multiset object keeps order, numerically. So, 
// I would assume that 6 will be positioned between 3 and 8, i.e.
// asnwer c. I'm a little confused by the selection for answers,
// I'm assuming you mean what spot in the array that 6 will take on,
// and I am saying it will take on the position of 8. At the end,
// the multiset object will look like: 1,3,6,8,11

// PROBLEM 2 C
// Answer B, p points to "2" in the array. [2] indicates to offset by 2.
// So p[2] points to "4".
 
// PROBLEM 3 A
template <class Item>
void stack<Item>::push(const Item& entry)
{
assert(size( ) < CAPACITY);
data[used] = entry;
++used;
}

// PROBLEM 3 B
template <class Item>
void stack<Item>::pop( )
{
assert(!empty( ));
--used;
}

// PROBLEM 4 (Using stack described in chapter 7)
#include "stack.h"
#include <iostream>
int main()
{
	//Overall Idea: Push userinput string into stacks 0,1
	// Pop 1 while storing into stack 2
	// Now we have two stacks that are ordered oppositely
	// For all things in stack, compare the tops respectively.
	// Start a counter that is the size of the stacks.
	// If top of both stacks is equal, decrement by 1
	// If at all the tops of the stacks don't equal, return

	// Initialize 3 stack objects
	stack<char> object0,object1,object2;
	// Variables for storing info
	string input;
	int i,length,numberThingsInStack;

	// Get string input
	printf("Enter string ...\n")
	getline(cin,str);
	length = str.size();

	// Push string into both object0 and object 1
	for(i = 0;i < length;i++)
	{
		object0.push(str[i]);
		object1.push(str[i]);
	}

	while(!object1.empty())
	{	
		object2.push(object1.top());
		object1.pop();
	}
	numberThingsInStack = object2.size();
	
	while(!object2.empty())
	{
		if(object0.top() == object2.top())
		{
			numberThingsInStack--;
			object0.pop();
			object1.pop();
		}
		else // i.e. they dont equal
		{
			cout << "user input is not a palindrome\n";
			return 0;
		}

	}

	cout << "user input is a palindrome\n";
	return 0
}