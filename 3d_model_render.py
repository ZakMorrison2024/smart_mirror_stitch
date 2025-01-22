from direct.showbase.ShowBase import ShowBase

class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Load the 3D model with animation
        self.model = self.loader.loadModel("animated_character.bam")
        self.model.reparentTo(self.render)

        # Play predefined animation (e.g. "walk")
        self.model.loop("walk")

app = MyApp()
app.run()
