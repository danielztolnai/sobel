# Main executable
TARGET = homework

# Directories
SRCDIR   = ./src
OBJDIR   = ./obj
BINDIR   = ./bin

# Files to compile
SOURCES  := $(wildcard $(SRCDIR)/*.c) $(wildcard $(SRCDIR)/*/*.c)
INCLUDES := $(wildcard $(SRCDIR)/*.h) $(wildcard $(SRCDIR)/*.h)
OBJECTS  := $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

# Compiler, linker, and flags
CC       = gcc
CFLAGS   = -Wall -I. -std=c11 -fopenmp -mavx -O3 -march=native -s -DNDEBUG -mtune=native
CFLAGSD  = -Wall -I. -std=c11 -fopenmp -mavx -O0 -march=native -v -da -Q -g
LFLAGS   = -Wall -lpng -lm -fopenmp -mavx

# Tasks
$(BINDIR)/$(TARGET): directories $(OBJECTS)
	@$(CC) $(OBJECTS) -o $@ $(LFLAGS)

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.c
	@$(CC) $(CFLAGS) -c $< -o $@

clean:
	@rm -f $(BINDIR)/* $(OBJDIR)/*

# Create directory structure
MKDIR_P = mkdir -p
.PHONY: directories
directories: $(OBJDIR) $(BINDIR)
$(OBJDIR):
	@$(MKDIR_P) $(OBJDIR)
$(BINDIR):
	@$(MKDIR_P) ${BINDIR}
