TARGET=sout


# can add : pdflatex -interaction=batchmode


all: $(TARGET)
	pdflatex -output-directory build -halt-on-error -file-line-error -halt-on-error $(TARGET).tex
	mv build/$(TARGET).pdf .

$(TARGET): build
	pdflatex -output-directory build -draftmode -halt-on-error -file-line-error $(TARGET).tex

build:
	mkdir -p build

.PHONY: clean mrproper

clean:
	rm -f build/*.aux build/*.log build/*.nav build/*.out build/*.snm build/*.toc
	$(shell echo "if [ -e build ] ; then rmdir build; fi")

mrproper: clean
	rm -f $(TARGET).pdf

