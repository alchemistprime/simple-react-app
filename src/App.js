import React from 'react'
import MessageList from './components/MessageList'
import SendMessageForm from './components/SendMessageForm'
//import RoomList from './components/RoomList'
// import NewRoomForm from './components/NewRoomForm' 

class App extends React.Component {
  render() {
      return (
          <div className="app">
              <MessageList />
              <SendMessageForm />
          </div>
      );
  }
}

export default App