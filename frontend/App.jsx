import React from 'react';
import UserList from './components/UserList';
import UserForm from './components/UserForm';

const App = () => {
    return (
        <div>
            <UserForm />
            <UserList />
        </div>
    );
};

export default App;