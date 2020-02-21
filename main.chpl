use gjallarbru;

var gj : gjallarbru.Gjallarbru = new shared gjallarbru.Gjallarbru();
gj.pInit();

// Basic stuff!
gj.runString("print('You know who is a string?  Me.  I am a string.')");
gj.runString("a = 12");
gj.runString("print(a)");

// Ooooh, look, a multiline string!
var a = "\
a = 24 \
print(a)";
gj.runString(a);

// Look ma, no 'please continue on the next line' indicators!
var b = """
a = 36
print(a)
print(a + 24)
print("oh man!")
""";
gj.runString(b);