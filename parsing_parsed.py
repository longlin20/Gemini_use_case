import pandas as pd

question1 = """In the store of the course project (StoreManager), there is a queue to store the orders to be processed (ordersToProcess). The queue is implemented in another class (LinkedQueue), whose main methods are provided below. In the queue, orders are of type Order, and are identified with the attribute orderID, which can be accessed with its corresponding getOrderID().\n\npublic class LinkedQueue {  \n private Node<Order> head; \n private Node<Order> tail; \n private int size; \n\n public LinkedQueue() {}  \n public boolean isEmpty() { ...} \n public int size() { ...} \n public Order front() { ...} \n public void enqueue(Order info) {...} \n public Order dequeue() { ...} \n public String toString() { ...} \n public void print() { ...} \n}"""

question2 = """In a project, we need a stack data structure, and we want to make use of ArrayLists for its implementation. Complete the code below to implement the main methods of the stack.\n\npublic class ArrayListStack<E> {  \n private ArrayList<E> arraylist ; \n\n public ArrayListStack(){  // COMPLETE } \n public void push(E info) { // COMPLETE  } \n public E pop() { // COMPLETE } \n}\n\nNOTE: The class ArrayList<E> has the following methods, some of which can be useful in this problem: \n• boolean add(E e) \n• boolean isEmpty() \n• void add(int index, E element) \n• E remove(int index) \n• void clear() \n• boolean remove(Object o) \n• E get(int index) \n• E set(int index, E element) \n• int indexOf(Object o) \n• int size()"""

question3 = """The Huffman coding is a typical encoder to encode information. This coding generates a tree based on the probability of each symbol (e.g., letter) and the encoding can be obtained based on the tree. An example of tree is as follows:  \n\nIn this tree, there a symbol (e.g., letter) in each leaf. In our implementation, we will represent them with String. The rest of the nodes are empty (in our case, we will put a dash, “-”, in the information in those nodes). In order to find the encoding, we add a “0” each time we move left in the tree and we add a “1” each time we move right in the tree. This way, the encoding of A is “00” (we move twice to the left) and the encoding of C is “10” (we move right first and then we move left)."""
data = {
    "question": [question1, question1, question1, question2, question3, question3, question3],
    "description": [
        "Considering the abovementioned information, implement the method Order search(int orderID) in class StoreManager. This method searches one order in the queue (given its orderID) and returns the corresponding Order object. For this method, you can only use the methods of class LinkedQueue that are already provided. Note that this method does not modify neither the content of the queue nor the order of its elements.",

        "The implementation of previous section would have been easier if there was a method to search in the LinkedQueue class. In this section, you are asked to implement the method Order search(int orderID) in LinkedQueue class. This method searches one order in the queue (given its orderID) and returns the corresponding Order object.",

        "Finally, implement the method Order search2(int orderID) in StoreManager. This method carries out the same operations of the method implemented in Section 1.1, but making use of the method implemented in Section 1.2.",

        "",

        "Indicate the encoding and the depth in the tree of the symbols B, D and E.",

        "Program the method String findEncoding(String symbol) in the class LBTree that represents the tree (you can find a partial implementation below). This method has a symbol as a parameter (e.g., “A”) and returns a String with the corresponding encoding (“00” in the example”). If the symbol is not found in the tree (e.g., if “F” is used in the previous tree), the method will return -1.\n\npublic class LBTree<E> implements BTree<E> {  \n private LBNode<E> root; \n\n public LBTree() {  \n  root = null; \n } \n\n public LBTree(E info) { \n  root = new LBNode<E>(info, new LBTree<E>(), new LBTree<E>());  \n } \n\n public boolean isEmpty() { ...} \n public E getInfo() throws BTreeException { ...} \n public BTree<E> getLeft() throws BTreeException { ...} \n public BTree<E> getRight() throws BTreeException { ...} \n public void insert(BTree<E> tree, int side) throws BTreeException { ...}  \n public BTree<E> extract(int side) throws BTreeException { ...} \n public int size() { ...} \n public String toString() { ...} \n public int height() { ...} \n public String findEncoding(String symbol) {// SECTION 3.2 } \n public int findDepth(String symbol) {// SECTION 3.3 } \n}",

        "Implement the method int findDepth(String symbol). This method indicates the depth of the node where a symbol is. If the symbol is not found, the method will return -1.\nHint: You can make use of the previous method."
    ],
    "ground_truth": [
        """public Order search( int orderID) { 
     Order result = null; 
     if(!ordersToProcess .isEmpty()) {  
      for(int i=0; i<ordersToProcess .size(); i++) { 
       Order aux = ordersToProcess .dequeue();  
       if(aux.getOrderID() == orderID) { 
        result = aux; 
       } 
       ordersToProcess .enqueue( aux); 
      } 
     } 
     return result; 
    }""",
        """public Order search( int orderID) { 
         Node<Order> aux = head; 
         while(aux != null) { 
          if(aux.getInfo().getOrderID()== orderID) { 
           return aux.getInfo();  
          } 
          aux = aux.getNext();  
         } 
         return null; 
    }""",
        """public Order search2( int orderID) { 
     return ordersToProcess .search( orderID); 
    }""",
        """public class ArrayListStack<E> {  
     private ArrayList<E> arraylist ; 

     public ArrayListStack(){  
      arraylist  = new ArrayList<E>();  
     } 

     public void push(E info) { 
      arraylist .add(0, info); // or arraylist.add(info)  
     } 

     public E pop() {  
      return arraylist .remove(0); // or arraylist.remove(arraylist.size())  
     } 
    }""",
        """Symbol  Encoding  Depth  
            B       01         2
            D       110        3 
            E       111        3 """,
        """public String findEncoding(String symbol) { 
     if(isEmpty()) { // Base case when reaching a leaf without finding the symbol  
      return "-1"; 
     } else if(root.getInfo().equals( symbol)) { // Base case when finding the 
    symbol 
      return ""; 
     } else { 
      // Add 0 when moving to the left  
      String left = "0" + root.getLeft().findEncoding( symbol); 
      // Add 1 when moving to the right  
      String right = "1" + root.getRight().findEncoding( symbol); 

      // If the left String does not contains -1,  
      // it means the symbol is found in the left part and it is returned  
      if(!left.contains( "-1")) return left; 
      // If the right String does not contains -1,  
      // it means the symbol is found in the left part and it is returned  
      if(!right.contains( "-1")) return right; 
      // If both left and right contain -1, it means the symbol is not found 
    and -1 is returned  
      return "-1"; 
     } 
    }""",
        """public int findDepth(String symbol) { 
     String coding = findEncoding( symbol); 
     if(coding.equals( "-1")) return -1; 
     return coding.length();  
    }"""
    ],
    "criteria": [
        """● 0.2: Create variable and return it at the end  
    ● 0.2: Loop to iterate over the size of the queue  
    ● 0.3: Deque element  
    ● 0.3: If to check if the orderIDs match.  
    ● 0.2: Assign the order to the result variable in case the orderID matches  
    ● 0.3: Enqueue element. If there is a break to stop enqueuing after finding the element, max 0.
    ● Significant errors are subject to additional penalties""",
        """● 0.2: Initialize aux node  
    ● 0.3: Loop to traverse the l ist (including update of aux)  
    ● 0.3: Check if the ID matches  
    ● 0.2: Return the information correctly  
    ● Significant errors are subject to additional penalties""",
        """● 0.5: Correct call to method search. If return is not used, max 0.2.  
    ● Significant errors are subject to additional penalties""",
        """● 0.2: Constructor  
    ● 0.4: Method push  
    ● 0.4: Method pop  
    ● Significant errors are subject to additional penalties""",
        """● Penalize 0.1 for each incorrect cell (e.g., 0.4 if 5/6 cells are correct and 0 if less than 2 cells are correct)  
    ● Significant errors are subject to additional penalties""",
        """● 0.3: Base case when reaching a leaf without finding the symbol  
    ● 0.3: Base case when finding the symbol  
    ● 0.4: Recursive case when moving to the left  
    ● 0.4: Recursive case when moving to the right  
    ● 0.6: Correct management to return the result depending on the part of the tree where symbol is found  
    ● Significant errors are subject to additional penalties""",
        """● 0.2: Call to findEncoding  
    ● 0.1: Check if coding is -1, and return -1 in that case  
    ● 0.2: Return the information when coding is not -1  
    ● Significant errors are subject to additional penalties"""
    ]
}


df = pd.DataFrame(data)
# Saving to Excel file
file_path = 'parsed_exam/exam2/2021_Midterm2.xlsx'
df.to_excel(file_path, index=False)
