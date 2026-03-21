CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

# Nombre del ejecutable
TARGET = matrix_multiplication

# Archivos fuente
SRCS = matrix_multiplication.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)