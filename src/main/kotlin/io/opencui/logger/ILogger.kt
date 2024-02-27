package io.opencui.logger

import io.opencui.core.Configuration
import io.opencui.core.ExtensionBuilder
import io.opencui.core.IExtension
import io.opencui.serialization.JsonElement
import java.sql.*
import java.time.LocalDateTime
import org.postgresql.util.PGobject

data class Turn(
    val utterance: String,
    val expectations: JsonElement,  // this should be an array of expectation, each is an object.
    val predictedFrameEvents: JsonElement,   // again an array of events.
    val dialogActs: JsonElement,    // an array of dialog acts.
    val timeStamp: LocalDateTime,
    val duTime: Long,
)  {
    var trueFrameEvents: JsonElement? = null  // this is provided manually when there are mistakes
    var dmTime: Long? = null // We might need this.
    var nluVersion: String? = null
    var duVersion: String? = null
    var channelType: String? = null
    var channelLabel: String? = null
    var userId: String? = null
}

interface ILogger: IExtension {
    fun log(turn: Turn): Boolean
}



data class Inserter(val stmt: PreparedStatement) {
    var index: Int = 1
    inline fun <reified T> add(value: T) {
        stmt.add(index, value)
        index += 1
    }

    inline fun <reified T> PreparedStatement.add(index: Int, v: T?) {
    if (v == null) {
        this.setNull(index, Types.NULL)
    } else {
        when (v) {
            is String -> this.setString(index, v)
            is Long -> this.setLong(index, v)
            is LocalDateTime -> this.setTimestamp(index, Timestamp.valueOf(v))
            is JsonElement -> this.setObject(index, PGobject().apply { type = "json"; value = v.toString() })
            else -> throw RuntimeException("not ready for this type")
        }
    }
}
}

data class JdbcLogger(val info: Configuration): ILogger {

    init {
        Class.forName(info[DRIVER]!! as String)
    }

    val conn : Connection by lazy {
        DriverManager.getConnection(info[URL] as String, info[USER] as String, info[PASSWORD] as String)
    }


    override fun log(turn: Turn): Boolean {
        val sqlStatement =
            """INSERT INTO logger.turn( \
               channel_type, channel_label, user_id, utterance, expectations, predicted_frame_events,\
               dialog_acts, time_stamp, true_frame_events, nlu_version, du_version, dm_time, du_time) \
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
   
        val builder = Inserter(conn.prepareStatement(sqlStatement));
        builder.add(turn.channelType)
        builder.add(turn.channelLabel)
        builder.add(turn.userId)
        builder.add(turn.utterance)
        builder.add(turn.expectations)
        builder.add(turn.predictedFrameEvents)
        builder.add(turn.dialogActs)
        builder.add(turn.timeStamp)
        builder.add(turn.trueFrameEvents)
        builder.add(turn.nluVersion)
        builder.add(turn.duVersion)
        builder.add(turn.dmTime)
        builder.add(turn.duTime)
        builder.stmt.executeUpdate()
        return true
    }


    companion object : ExtensionBuilder<ILogger> {
        const val TABLE : String = "turn"
        const val URL: String = "pgUrl"
        const val USER: String = "adminEmail"
        const val PASSWORD: String = "adminPassword"
        const val DRIVER: String = "driver"

        override fun invoke(p1: Configuration): ILogger {
            return JdbcLogger(p1)
        }

    }
}