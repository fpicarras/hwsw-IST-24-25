# Makefile for simple_cnn project

# Compiler and flags
CC = gcc
CFLAGS = -Wall -O2

# Object directory
OBJDIR = obj

# Source files and target
SRCS = simple_cnn.c image.c gemm.c
OBJS = $(SRCS:.c=.o)
OBJFILES = $(addprefix $(OBJDIR)/, $(OBJS))
TARGET = simple_cnn

# Libraries (if any)
LIBS = -lm

# Default target
all: $(OBJDIR) $(TARGET)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(TARGET): $(OBJFILES)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

$(OBJDIR)/%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(TARGET)

.PHONY: all clean