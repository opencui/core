package io.opencui.test

import com.fasterxml.jackson.annotation.JsonIgnore
import io.opencui.core.*
import io.opencui.core.Annotation
import io.opencui.core.da.UserDefinedInform
import kotlin.reflect.KMutableProperty0


public data class HelloWorldService(
  public override var session: UserSession? = null
) : IIntent {
  @JsonIgnore
  public override val type: FrameKind = FrameKind.BIGINTENT

  @get:JsonIgnore
  public val component_0915: IComponent_0915
    public get() = session!!.getExtension<IComponent_0915>()!!

  @JsonIgnore
  public override var annotations: Map<String, List<Annotation>> = mutableMapOf()

  public override fun searchResponse(): Action? = when {
    else -> HelloWorld_else_branch(this)
  }

  public override fun createBuilder(p: KMutableProperty0<out Any?>?): FillBuilder = object :
      FillBuilder {
    public var frame: HelloWorldService? = this@HelloWorldService

    public override fun invoke(path: ParamPath): FrameFiller<HelloWorldService> {
      val filler = FrameFiller({(p as? KMutableProperty0<HelloWorldService?>) ?: ::frame}, path)
      return filler
    }
  }

  public inner class HelloWorld_else_branch(
    public val frame: HelloWorldService
  ) : SeqAction(
      listOf(
          TextOutputAction {
              UserDefinedInform(
                  this, "me.test.testApp_1012.HelloWorld",
                  simpleTemplates(
                      with(frame) {
                          """component_0915.testFuncton：${component_0915.testFunction(":)")}""".trimMargin()
                      }
                  )
              )
          }
      )
  )
}

