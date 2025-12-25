VERSION = 2.3.4

APP0 = sugar
BINDIR = /usr/local/bin
LIBDIR = /usr/local/lib/$(APP0)

APP = $(APP0)-$(VERSION)
JAR = ./build/$(APP).jar
ZIP = ./build/$(APP).zip

JAVA_VERSION = 1.8
# JAVA_RT = /usr/lib/jvm/java-8-openjdk-amd64/jre/lib/rt.jar

WEBPAGE = https://cspsat.gitlab.io/sugar/
WEBTITLE = Sugar: A SAT-based Constraint Solver
PACKAGE = jp.kobe_u.sugar

DOCTITLE = Sugar version $(VERSION) Core API Specification
JAVADOC  = javadoc -locale en_US \
	-windowtitle '$(DOCTITLE)' -doctitle '$(DOCTITLE)' \
	-header '<em><b>$(DOCTITLE)</b></em><br>' \
	-bottom '<address><a href="$(WEBPAGE)">$(WEBTITLE)</a></address>' \
	-link http://java.sun.com/javase/8/docs/api \
	-classpath ./classes -sourcepath ./src -subpackages $(PACKAGE) \
	-public -d ./docs/api

all: javac jar javadoc zip

javac:
	javac -target $(JAVA_VERSION) -source $(JAVA_VERSION) \
		-sourcepath ./src -d ./classes \
		./src/jp/kobe_u/sugar/*.java \
		./src/jp/kobe_u/sugar/*/*.java

# XXXjavac:
# 	javac -target $(JAVA_VERSION) -source $(JAVA_VERSION) -bootclasspath $(JAVA_RT) \
# 		-sourcepath ./src -d ./classes \
# 		./src/jp/kobe_u/sugar/*.java \
# 		./src/jp/kobe_u/sugar/*/*.java

jar:
	jar cmf manifest $(JAR) -C ./classes .

javadoc:
	rm -rf ./docs/api
	LC_ALL=C LANG=C $(JAVADOC) $(PACKAGE)

zip:
	rm -f $(ZIP)
	rm -rf $(APP)
	mkdir $(APP)
	cp -pr ./src ./classes ./docs ./examples ./tools $(APP)
	mkdir $(APP)/bin
	cp -p $(APP0) Makefile manifest $(JAR) $(APP)/bin
	find $(APP) \( -name .svn -o -name CVS -o -name .cvsignore -o -name '*~' \) -exec rm -rf '{}' '+'
	zip -q -r $(ZIP) $(APP)
	rm -rf $(APP)

install:
	mkdir -p $(LIBDIR)
	cp -p $(JAR) $(LIBDIR)
	mkdir -p $(BINDIR)
	cp -p $(APP0) $(APP)
	install $(APP) $(BINDIR)
	rm $(APP)
	cd $(BINDIR); rm -f $(APP0); ln -s $(APP) $(APP0)

clean:
	rm -rf ./docs/api
	rm -rf $(ZIP) $(JAR)

