<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.individular.hotnews-full</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>__REFRESH_SCRIPT__</string>
        <string>full</string>
    </array>
    <key>WorkingDirectory</key>
    <string>__PROJECT_DIR__</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>__UV_PATH__</string>
        <key>HOME</key>
        <string>__USER_HOME__</string>
    </dict>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>9</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>RunAtLoad</key>
    <false/>
    <key>StandardOutPath</key>
    <string>__LOG_DIR__/launchd-full.out.log</string>
    <key>StandardErrorPath</key>
    <string>__LOG_DIR__/launchd-full.err.log</string>
</dict>
</plist>