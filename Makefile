#-------------------------------------------
DEBUG_EN=1
APP=test_app
TOP_DIR=$(shell pwd)
ARCH=
CC=gcc
CFLAGS= -Wall -Wfatal-errors -std=c99
LDFLAGS=
SRC_TOP=./src
OBJ_DIR=./obj/
INC_DIRS=./inc
LIB_DIRS=


ifeq ($(DEBUG_EN), 1) 
CFLAGS+=-O0 -g -DDEBUG_EN
endif
# Dependencies if any
LIBS= m openblas
DEPS=$(wildcard ./inc/*.h)

CFLAGS+= $(foreach D,$(INC_DIRS),-I$D)
LDFLAGS+= $(foreach D,$(LIB_DIRS),-L$D) $(foreach L,$(LIBS), -l$(L))


#-------------------------------------------
# Source files for compilation
#-------------------------------------------

SRCS=$(wildcard $(SRC_TOP)/*.c)

# List of all objects that are to be built
OBJS= $(addprefix $(OBJ_DIR), $(patsubst %.c, %.o, $(notdir $(SRCS))))

vpath %.c $(dir $(SRCS))

#-------------------------------------------
# Build targets
all : obj $(APP) $(DEPS)

$(APP): Makefile $(OBJ_DIR) $(OBJS)
	@echo "Linking..."
	$(CC)  -o $@  $(OBJS) $(LDFLAGS)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)
	
$(OBJS):$(OBJ_DIR)%.o:%.c
	@echo "Building $@ from $<..."
	$(CC) -c $(CFLAGS) $< -o $@
	
#-------------------------------------------

#-------------------------------------------
# Phony targets
#-------------------------------------------
.phony: clean clean_all

obj :
	mkdir -p obj

# Clean the common object files
clean:
	@echo "Removing all objects..."
	rm -rf $(OBJS)
	rm -f $(APP)
	
# Clean all libraries if at all
clean_all:
	@echo "Cleaning everything..."

# Print important variables and values
print:
	@echo "Compiler flags = $(CFLAGS)"
	@echo "Linker flags = $(LDFLAGS)"
	@echo "Source files = $(SRCS)"
	@echo "Object files = $(OBJS)"
	@echo "Dependencies = $(DEPS)"

